import cv2
import time
import torch
from ultralytics import YOLO

print("CUDA Available:", torch.cuda.is_available())

# ==========================================================
# LOAD YOLO MODEL
# ==========================================================
best_model = YOLO("../config_files/best.pt")

# ==========================================================
# CAMERA
# ==========================================================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

# ==========================================================
# STATE MACHINE
# ==========================================================
state = "LIVE"

captured_frame = None

label = "Unknown"

confidence = 0.0

# ==========================================================
# TIMERS
# ==========================================================
capture_delay = 0.8

display_duration = 3

capture_start = 0

display_start = 0

# ==========================================================
# CENTER BOX SETTINGS
# ==========================================================
BOX_WIDTH = 300
BOX_HEIGHT = 300

# ==========================================================
# MAIN LOOP
# ==========================================================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    frame_height, frame_width = frame.shape[:2]

    # ======================================================
    # CENTER BOX COORDINATES
    # ======================================================
    x1 = (frame_width // 2) - (BOX_WIDTH // 2)
    y1 = (frame_height // 2) - (BOX_HEIGHT // 2)

    x2 = x1 + BOX_WIDTH
    y2 = y1 + BOX_HEIGHT

    # ======================================================
    # LIVE STATE
    # ======================================================
    if state == "LIVE":

        display_frame = frame.copy()

        # --------------------------------------------------
        # DRAW CENTER BOX
        # --------------------------------------------------
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            display_frame,
            "Place Waste Item Inside Box",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            "Press SPACE To Capture",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "YOLO Waste Classification",
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # --------------------------------------------------
        # SPACE KEY CAPTURE
        # --------------------------------------------------
        if key == 32:

            capture_start = current_time

            state = "WAIT"

        # ESC KEY
        elif key == 27:
            break

    # ======================================================
    # WAIT STATE
    # ======================================================
    elif state == "WAIT":

        display_frame = frame.copy()

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3
        )

        cv2.putText(
            display_frame,
            "Stabilizing...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "YOLO Waste Classification",
            display_frame
        )

        if current_time - capture_start >= capture_delay:

            captured_frame = frame.copy()

            state = "CAPTURED"

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ======================================================
    # CAPTURED STATE
    # ======================================================
    elif state == "CAPTURED":

        frame_copy = captured_frame.copy()

        # --------------------------------------------------
        # CROP CENTER BOX REGION
        # --------------------------------------------------
        crop = frame_copy[y1:y2, x1:x2]

        label = "Unknown"

        confidence = 0.0

        # --------------------------------------------------
        # YOLO CLASSIFICATION
        # --------------------------------------------------
        results = best_model.predict(
            source=crop,
            imgsz=224,
            conf=0.25,
            verbose=False
        )

        probs = results[0].probs

        names = results[0].names

        if probs is not None:

            top1 = probs.top1

            confidence = probs.top1conf.item()

            if confidence >= 0.35:
                label = names[top1]
            else:
                label = "Unknown"

        # --------------------------------------------------
        # DRAW RESULT BOX
        # --------------------------------------------------
        cv2.rectangle(
            frame_copy,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame_copy,
            f"{label}: {confidence:.2f}",
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        captured_frame = frame_copy

        display_start = current_time

        state = "DISPLAY"

    # ======================================================
    # DISPLAY STATE
    # ======================================================
    elif state == "DISPLAY":

        cv2.imshow(
            "YOLO Waste Classification",
            captured_frame
        )

        if current_time - display_start >= display_duration:

            state = "LIVE"

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

# ==========================================================
# CLEANUP
# ==========================================================
cap.release()

cv2.destroyAllWindows()