import cv2 as cv
import av
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model
np_model = YOLO('yolo11n.pt')

# Load the video stream
container = av.open('rtsp://admin:@192.168.1.10:554/stream1')

# Set the detection interval in seconds
detection_interval = 1.0  # Run detection every 1 second

last_detection_time = time.time()
last_detections = None

for packet in container.demux(video=0):
    for frame in packet.decode():
        img = frame.to_ndarray(format='bgr24')

        current_time = time.time()
        # Check if it's time to run detection
        if current_time - last_detection_time >= detection_interval:
            # Run YOLO detection
            results = np_model(img)
            last_detections = results
            last_detection_time = current_time

        # Draw the last detections on the current frame
        if last_detections is not None:
            for result in last_detections:
                boxes = result.boxes.xyxy
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Display the video frame
        cv.imshow('Video Stream', img)

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()