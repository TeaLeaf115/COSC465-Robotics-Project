import cv2
import time
import torch
from ultralytics import YOLO
from picamera2 import Picamera2
from flask import Flask, Response

app = Flask(__name__)

print("CUDA Available:", torch.cuda.is_available())

# ==========================================================
# LOAD YOLO MODEL
# ==========================================================
best_model = YOLO("../config_files/best.pt")

# ==========================================================
# CAMERA
# ==========================================================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
picam2.start()

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
# VIDEO FEED GENERATOR
# ==========================================================
def video_feed():
    global state, captured_frame, label, confidence
    
    while True:
        frame = picam2.capture_array()

        current_time = time.time()
        
        frame_height, frame_width, _ = frame.shape
        
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
            
            if current_time - capture_start >= capture_delay:

                captured_frame = frame.copy()
                
                state = "CAPTURED"
                
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

            display_frame = frame_copy

            display_start = current_time
            
            state = "DISPLAY"

        # ======================================================
        # DISPLAY STATE
        # ======================================================
        elif state == "DISPLAY":
            
            if current_time - display_start >= display_duration:

                state = "LIVE"
            
        # ======================================================
        # ENCODE FRAME TO JPEG
        # ======================================================
        _, encoded_frame = cv2.imencode('.jpg', display_frame)
        
        # ======================================================
        # YIELD FRAME TO FLASK
        # ======================================================
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')

# ==========================================================
# FLASK ROUTES
# ==========================================================
@app.route('/')
def index():
    return Response(video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================================
# MAIN
# ==========================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)