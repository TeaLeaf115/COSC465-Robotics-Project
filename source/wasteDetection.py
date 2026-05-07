import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import time
import numpy as np

# ----------------------------
# Motion Detection
# ----------------------------
def detect_motion(frame1, frame2, threshold=25):

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    _, thresh = cv2.threshold(
        diff,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cpu")

# ----------------------------
# RESNET18 MODEL
# ----------------------------
num_classes = 3

model = resnet18(weights=ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

model.load_state_dict(
    torch.load(
        "config_files/RESNET18_ColorJitter_CNN.pth",
        map_location=device
    )
)

model.eval()

# ----------------------------
# CLASS NAMES
# ----------------------------
classNames = [
    "Compost",
    "Recycle",
    "Trash"
]

# ----------------------------
# TRANSFORM
# ----------------------------
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

net = cv2.dnn_DetectionModel(
    weightsPath,
    configPath
)

net.setInputSize(320, 320)

net.setInputScale(1.0 / 127.5)

net.setInputMean((127.5, 127.5, 127.5))

net.setInputSwapRB(True)

thres = 0.5

# ----------------------------
# STATE MACHINE
# ----------------------------
state = "LIVE"

captured_frame = None

label = "Unknown"

# ----------------------------
# TIMERS
# ----------------------------
capture_delay = 0.8

display_duration = 3

capture_start = 0

display_start = 0

# ----------------------------
# FRAME SKIPPING
# ----------------------------
frame_count = 0

# ----------------------------
# CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame1 = cap.read()

ret, frame2 = cap.read()

# ----------------------------
# MAIN LOOP
# ----------------------------
while cap.isOpened():

    frame_count += 1

    # ----------------------------
    # Skip Frames For Speed
    # ----------------------------
    if frame_count % 2 != 0:

        frame1 = frame2

        ret, frame2 = cap.read()

        continue

    current_time = time.time()

    # ==========================================================
    # LIVE
    # ==========================================================
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

        cv2.putText(
            frame1,
            "Live Feed",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Waste Detection",
            frame1
        )

    # ==========================================================
    # WAIT
    # ==========================================================
    elif state == "WAIT":

        if current_time - capture_start >= capture_delay:

            captured_frame = frame1.copy()

            state = "CAPTURED"

        cv2.putText(
            frame1,
            "Stabilizing...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Waste Detection",
            frame1
        )

    # ==========================================================
    # CAPTURED
    # ==========================================================
    elif state == "CAPTURED":

        frame = captured_frame.copy()

        label = "Unknown"

        # ---------------------------------
        # Resize For Faster SSD Detection
        # ---------------------------------
        small = cv2.resize(frame, (320, 320))

        classIds, confs, bbox = net.detect(
            small,
            confThreshold=thres
        )

        if len(classIds) != 0:

            for i in range(len(classIds)):

                # skip people
                if classIds[i] == 1:
                    continue

                x, y, w, h = bbox[i]

                # ---------------------------------
                # Scale Back To Original Frame
                # ---------------------------------
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 320

                x = int(x * scale_x)
                y = int(y * scale_y)
                w = int(w * scale_x)
                h = int(h * scale_y)

                # ---------------------------------
                # Padding Around Crop
                # ---------------------------------
                padding = 20

                x1 = max(0, x - padding)
                y1 = max(0, y - padding)

                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # ---------------------------------
                # BGR -> RGB
                # ---------------------------------
                crop = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB
                )

                # ---------------------------------
                # TRANSFORM
                # ---------------------------------
                input_tensor = transform(crop).unsqueeze(0).to(device)

                # ---------------------------------
                # MODEL INFERENCE
                # ---------------------------------
                with torch.no_grad():

                    outputs = model(input_tensor)

                    probs = torch.softmax(outputs, dim=1)

                    conf, pred = torch.max(probs, 1)

                # ---------------------------------
                # CONFIDENCE FILTER
                # ---------------------------------
                if conf.item() < 0.6:
                    continue

                label = classNames[pred.item()]

                # ---------------------------------
                # DRAW BOX
                # ---------------------------------
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{label}: {conf.item():.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                break

        captured_frame = frame

        display_start = current_time

        state = "DISPLAY"

    # ==========================================================
    # DISPLAY
    # ==========================================================
    elif state == "DISPLAY":

        frame = captured_frame.copy()

        cv2.putText(
            frame,
            label,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        cv2.imshow(
            "Waste Detection",
            frame
        )

        if current_time - display_start >= display_duration:

            state = "LIVE"

    # ==========================================================
    # FRAME UPDATE
    # ==========================================================
    frame1 = frame2

    ret, frame2 = cap.read()

    if not ret:
        break

    # ESC KEY
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()

cv2.destroyAllWindows()