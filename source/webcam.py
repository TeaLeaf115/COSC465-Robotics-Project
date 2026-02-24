import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# ----------------------------
# Load Your PyTorch Model
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define same model architecture
class ConvolutionNNET(nn.Module):
    def __init__(self):
        super(ConvolutionNNET, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.features(x)
model_path = "config_files/waste_model30Epochs.pth"
model = ConvolutionNNET().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

classNames = ['Compost', 'Recycle', 'Trash']

# Image preprocessing (must match training)
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
# Load Detection Model (SSD)
# ----------------------------

configPath = "config_files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "config_files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

thres = .63
nms_threshold = 0.5

# ----------------------------
# Start Webcam
# ----------------------------

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    if len(classIds) != 0:
        bbox = list(bbox)
        confs = list(map(float, confs.flatten()))
        classIds = classIds.flatten()

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = bbox[i]

                # Crop detected region
                crop = image[y:y+h, x:x+w]

                if crop.size == 0:
                    continue

                # Preprocess for PyTorch model
                input_tensor = transform(crop).unsqueeze(0).to(device)

                # Classify
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    label = classNames[predicted.item()]

                # Draw bounding box
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Put your waste class label
                cv2.putText(
                    image,
                    label,
                    (x + 10, y + 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Waste Detection", image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()