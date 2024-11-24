# Import required libraries
import cv2 as cv  # OpenCV for image processing
import av  # PyAV for video handling
import numpy as np  # NumPy for numerical operations
from ultralytics import YOLO  # YOLO object detection model
import time  # For timing operations
import logging  # For controlling log output
from image_api import print_license_plate  # Custom module for license plate processing
import threading  # For running license plate processing in parallel

def iou(box1, box2):
    """ 
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Used to determine if two detected boxes represent the same car.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
    Returns:
        float: IoU score between 0 and 1
    """
    # Calculate coordinates of intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate area of intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    return inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0

# Suppress unnecessary logging messages from ultralytics
logging.getLogger('ultralytics').setLevel(logging.NOTSET)

# Initialize YOLO model for object detection
np_model = YOLO('yolo11n.pt')

# Get the model's class names dictionary
class_names = np_model.names

# Find class IDs that correspond to 'car' in the model
car_class_ids = [cls_id for cls_id, name in class_names.items() if name.lower() == 'car']

# Exit if 'car' class is not found in the model
if not car_class_ids:
    print("Class 'car' not found in the model's class names.")
    exit()

# Initialize video stream from RTSP camera
container = av.open('rtsp://admin:@192.168.1.10:554/stream1')

# Configuration parameters
detection_interval = 1.0  # Time between detection runs in seconds

# Initialize tracking variables
last_detection_time = time.time()
last_detections = None
best_confidence = 0.0
best_car_img_masked = None
detected_boxes = []  # List to track previously detected cars

# Main video processing loop
for packet in container.demux(video=0):
    for frame in packet.decode():
        # Convert frame to OpenCV format
        img = frame.to_ndarray(format='bgr24')

        current_time = time.time()
        # Run detection at specified intervals
        if current_time - last_detection_time >= detection_interval:
            # Perform object detection
            results = np_model(img)
            last_detections = results
            last_detection_time = current_time

            # Initialize frame-specific tracking variables
            car_detected_in_frame = False
            max_confidence_in_frame = 0.0
            current_car_id = None
            current_detected_boxes = []

            if last_detections is not None:
                # Process each detection in the frame
                for result in last_detections:
                    boxes = result.boxes.xyxy  # Bounding box coordinates
                    class_ids = result.boxes.cls  # Detected class IDs
                    confidences = result.boxes.conf  # Detection confidence scores
                    
                    # Process each detected object
                    for box, cls_id, conf in zip(boxes, class_ids, confidences):
                        if int(cls_id) in car_class_ids:
                            current_detected_boxes.append(box)
                            car_detected_in_frame = True

                            # Check if this is a new car by comparing with previous detections
                            is_new_car = True
                            for prev_box in detected_boxes:
                                if iou(prev_box, box) > 0.5:  # Using IoU threshold of 0.5
                                    is_new_car = False
                                    break
                            
                            # Handle new car detection
                            if is_new_car:
                                print("Nuevo carro detectado")
                                best_confidence = 0.0
                                best_car_img_masked = None

                            # Process if it's a new car with higher confidence
                            if is_new_car and conf > max_confidence_in_frame:
                                max_confidence_in_frame = conf
                                x1, y1, x2, y2 = map(int, box)

                                # Extract car image from frame
                                car_img = img[y1:y2, x1:x2].copy()
                                
                                # Create rounded corner mask (currently unused)
                                height, width = car_img.shape[:2]
                                radius = int(min(width, height) / 10)
                                mask = np.zeros((height, width), dtype=np.uint8)
                                cv.rectangle(mask, (0, 0), (width, height), 255, -1)
                                cv.circle(mask, (radius, radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, radius), radius, 0, -1)
                                cv.circle(mask, (radius, height - radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, height - radius), radius, 0, -1)

                                best_car_img_masked = car_img

                # Update best detection if needed
                if car_detected_in_frame and max_confidence_in_frame > best_confidence:
                    best_confidence = max_confidence_in_frame
                
                    # Save the best car image
                    filename = 'car.png'
                    print(f"Guardando con confianza {best_confidence:.2f}")
                
                    if best_car_img_masked is not None and best_car_img_masked.size > 0:
                        cv.imwrite(filename, best_car_img_masked)
                        # Process license plate in a separate thread
                        threading.Thread(target=print_license_plate).start()
                    else:
                        print("Error guardando la imagen")

            # Update tracking of detected boxes
            if current_detected_boxes is not None:
                detected_boxes = current_detected_boxes

        # Draw detection boxes on the display frame
        if last_detections is not None:
            for result in last_detections:
                boxes = result.boxes.xyxy
                class_ids = result.boxes.cls
                for box, cls_id in zip(boxes, class_ids):
                    if int(cls_id) in car_class_ids:
                        x1, y1, x2, y2 = map(int, box)
                        # Draw green rectangle around detected car
                        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Add 'Car' label above the box
                        cv.putText(img, 'Car', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the processed frame
        cv.imshow('Video Stream', img)

        # Check for 'q' key press to exit
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

# Clean up OpenCV windows
cv.destroyAllWindows()