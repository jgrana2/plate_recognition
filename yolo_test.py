import cv2 as cv
import av
import numpy as np
from ultralytics import YOLO
import time
import logging

# Set the logging level for 'ultralytics' to WARNING to suppress INFO messages
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Load the YOLOv8 model
np_model = YOLO('yolo11n.pt')

# Get the class names
class_names = np_model.names

# Find the class IDs for 'car'
car_class_ids = [cls_id for cls_id, name in class_names.items() if name.lower() == 'car']

if not car_class_ids:
    print("Class 'car' not found in the model's class names.")
    exit()

# Load the video stream
container = av.open('rtsp://admin:@192.168.1.10:554/stream1')

# Set the detection interval in seconds
detection_interval = 1.0  # Run detection every 1 second

last_detection_time = time.time()
last_detections = None

# Variables to store the best detection
best_confidence = 0.0
best_car_img_masked = None

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

            # Flag to check if any car is detected in this frame
            car_detected_in_frame = False

            # Process detections to find the best one
            if last_detections is not None:
                max_confidence_in_frame = 0.0  # Reset for each frame
                for result in last_detections:
                    boxes = result.boxes.xyxy
                    class_ids = result.boxes.cls
                    confidences = result.boxes.conf
                    # Filter detections for 'car' class only
                    for box, cls_id, conf in zip(boxes, class_ids, confidences):
                        if int(cls_id) in car_class_ids:
                            car_detected_in_frame = True  # Car detected in this frame
                            if conf > max_confidence_in_frame:
                                max_confidence_in_frame = conf
                                x1, y1, x2, y2 = box
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                # Extract the car image from the frame
                                car_img = img[y1:y2, x1:x2].copy()
                                # Create a roundbox mask
                                height, width = car_img.shape[:2]
                                radius = int(min(width, height) / 10)  # Adjust the radius as needed
                                # Create a mask with rounded corners
                                mask = np.zeros((height, width), dtype=np.uint8)
                                cv.rectangle(mask, (0, 0), (width, height), 255, -1)
                                cv.circle(mask, (radius, radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, radius), radius, 0, -1)
                                cv.circle(mask, (radius, height - radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, height - radius), radius, 0, -1)
                                # Apply the mask to the car image
                                best_car_img_masked = cv.bitwise_and(car_img, car_img, mask=mask)

                # Update the best confidence with the frame's max
                best_confidence = max_confidence_in_frame

            # If no car is detected in this frame, reset the best detection
            if not car_detected_in_frame:
                best_confidence = 0.0
                best_car_img_masked = None

        # Draw the last detections on the current frame
        if last_detections is not None:
            for result in last_detections:
                boxes = result.boxes.xyxy
                class_ids = result.boxes.cls
                # Filter detections for 'car' class only
                for box, cls_id in zip(boxes, class_ids):
                    if int(cls_id) in car_class_ids:
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # Draw bounding box on the main image
                        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Optionally, display the class name
                        cv.putText(img, 'Car', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the video frame
        cv.imshow('Video Stream', img)

        # Display the best car image in a separate window if available
        if best_car_img_masked is not None:
            cv.imshow('Best Detected Car', best_car_img_masked)
        else:
            # Close the 'Best Detected Car' window if no car is detected
            try:
                cv.destroyWindow('Best Detected Car')
            except cv.error:
                pass  # Window is already closed or doesn't exist

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# After the loop ends, save the best detected car image if not already saved
if best_car_img_masked is not None:
    cv.imwrite('best_car_image.png', best_car_img_masked)

cv.destroyAllWindows()