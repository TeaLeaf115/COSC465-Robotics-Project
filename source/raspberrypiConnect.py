import serial
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import time
import numpy as np


### Serial Connection Setup ###
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

max_retries = 10
try_count = 0
while not ser.is_open:
    if try_count >= max_retries:
        print("Failed to establish serial connection after multiple attempts.")
        exit(1)
    print("Waiting for serial connection...")
    try_count += 1
    time.sleep(1)
time.sleep(2)  # wait for the serial connection to initialize
ser.reset_input_buffer()
print("Serial connection established")



def send_label(command):
    if ser.is_open:
        ser.write(command.encode())
        print(f"Sent command: {command}")
        return "Command sent successfully"
    else:
        print("Serial connection is not open. Cannot send command.")
        return "Failed to send command: Serial connection not open"

# ----------------------------
# Motion Detection
# ----------------------------
def detect_motion(frame1, frame2, threshold=25):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# ----------------------------
# CNN MODEL
# ----------------------------
device = torch.device("cpu")

num_classes = 3
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
model.load_state_dict(torch.load("config_files/RESNET18_CNN30Epochs.pth", map_location=device))
model.eval()

classNames = ['Compost', 'Recycle', 'Trash']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------------------
# SSD MODEL
# ----------------------------
configPath = "config_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "config_files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

thres = 0.4

# ----------------------------
# STATE MACHINE
# ----------------------------
state = "LIVE"   # LIVE → WAIT → CAPTURED → DISPLAY
captured_frame = None
label = "Unknown"

# timing
capture_delay = 0.5
display_duration = 3

capture_start = 0
display_start = 0

# frame skipping
frame_count = 0

# ----------------------------
# CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # lower res = faster
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame1 = cap.read()
ret, frame2 = cap.read()




# ----------------------------
# MAIN LOOP
# ----------------------------
while cap.isOpened():

    frame_count += 1

    # skip frames for speed
    if frame_count % 2 != 0:
        frame1 = frame2
        ret, frame2 = cap.read()
        continue

    current_time = time.time()

    # ----------------------------
    # LIVE
    # ----------------------------
    if state == "LIVE":

        contours = detect_motion(frame1, frame2)
        motion_detected = False

        for c in contours:
            if cv2.contourArea(c) < 2000:
                continue
            motion_detected = True

        if motion_detected:
            capture_start = current_time
            state = "WAIT"

        cv2.putText(frame1, "Live Feed", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Waste Detection", frame1)

    # ----------------------------
    # WAIT (non-blocking delay)
    # ----------------------------
    elif state == "WAIT":

        if current_time - capture_start >= capture_delay:
            captured_frame = frame1.copy()
            state = "CAPTURED"

        cv2.putText(frame1, "Stabilizing...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Waste Detection", frame1)

    # ----------------------------
    # CAPTURED (run model once)
    # ----------------------------
    elif state == "CAPTURED":

        frame = captured_frame.copy()
        label = "Unknown"

        # 🔥 resize for faster detection
        small = cv2.resize(frame, (320, 320))

        classIds, confs, bbox = net.detect(small, confThreshold=thres)

        if len(classIds) != 0:
            for i in range(len(classIds)):

                if classIds[i] == 1:  # skip person
                    continue

                x, y, w, h = bbox[i]

                # scale back to original size
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 320

                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)

                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    continue

                input_tensor = transform(crop).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                if conf.item() < 0.7:
                    continue

                label = classNames[pred.item()]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break

        captured_frame = frame
        display_start = current_time
        state = "DISPLAY"

    # ----------------------------
    # DISPLAY (show result)
    # ----------------------------
    elif state == "DISPLAY":

        frame = captured_frame.copy()

        cv2.putText(frame, label, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Waste Detection", frame)

        if label and label != "Unknown":
            send_label(label)
            
        if current_time - display_start >= display_duration:
            state = "LIVE"

    # ----------------------------
    # FRAME UPDATE
    # ----------------------------
    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
ser.close()