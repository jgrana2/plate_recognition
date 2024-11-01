import cv2 as cv
import av
import numpy as np
from ultralytics import YOLO
import time
import logging
from image_api import print_license_plate
import threading

def iou(box1, box2):
    """ Calculate the Intersection over Union (IoU) between two bounding boxes. """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0

# Set the logging level for 'ultralytics' to WARNING to suppress INFO messages
logging.getLogger('ultralytics').setLevel(logging.NOTSET)

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

# Variables to store the best detection information
best_confidence = 0.0
best_car_img_masked = None
detected_boxes = []  # Track detected boxes across frames

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

            # Flag to check if any car was detected in this frame
            car_detected_in_frame = False
            max_confidence_in_frame = 0.0  # To track maximum confidence in this frame
            current_car_id = None  # Track the currently detected car ID
            current_detected_boxes = []  # Store bounding boxes of current detections

            if last_detections is not None:
                for result in last_detections:
                    boxes = result.boxes.xyxy
                    class_ids = result.boxes.cls
                    confidences = result.boxes.conf
                    
                    for box, cls_id, conf in zip(boxes, class_ids, confidences):
                        if int(cls_id) in car_class_ids:
                            current_detected_boxes.append(box)  # Add current box to the list
                            car_detected_in_frame = True  # Set flag if car is detected

                            # New car detection check
                            is_new_car = True
                            for prev_box in detected_boxes:  # Check against previous detections
                                if iou(prev_box, box) > 0.5:  # Using IoU threshold
                                    is_new_car = False
                                    break
                            
                            if is_new_car:
                                print("Nuevo carro detectado")
                                best_confidence = 0.0
                                best_car_img_masked = None
                            # else:
                                # print("Same old car")

                            if is_new_car and conf > max_confidence_in_frame:
                                max_confidence_in_frame = conf
                                x1, y1, x2, y2 = map(int, box)

                                # Extract the car image from the frame
                                car_img = img[y1:y2, x1:x2].copy()
                                
                                # Create a roundbox mask
                                height, width = car_img.shape[:2]
                                radius = int(min(width, height) / 10)  # Adjust the radius as needed
                                mask = np.zeros((height, width), dtype=np.uint8)
                                cv.rectangle(mask, (0, 0), (width, height), 255, -1)
                                cv.circle(mask, (radius, radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, radius), radius, 0, -1)
                                cv.circle(mask, (radius, height - radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, height - radius), radius, 0, -1)

                                # Apply the mask to the car image
                                best_car_img_masked = cv.bitwise_and(car_img, car_img, mask=mask)

                # Only update best confidence if a car was detected
                if car_detected_in_frame:
                    if max_confidence_in_frame > best_confidence:
                        best_confidence = max_confidence_in_frame
                
                        # Save the best car image immediately
                        filename = 'car.png'
                        print(f"Guardando con confianza {best_confidence:.2f}")
                
                        # Check if the image is not empty
                        if best_car_img_masked is not None and best_car_img_masked.size > 0:
                            cv.imwrite(filename, best_car_img_masked)
                        else:
                            print("Error guardando la imagen")
                        
                        threading.Thread(target=print_license_plate).start()                

            # Update detected boxes for the next frame
            if current_detected_boxes is not None:
                detected_boxes = current_detected_boxes

        # Draw the last detections on the current frame
        if last_detections is not None:
            for result in last_detections:
                boxes = result.boxes.xyxy
                class_ids = result.boxes.cls
                # Filter detections for 'car' class only
                for box, cls_id in zip(boxes, class_ids):
                    if int(cls_id) in car_class_ids:
                        x1, y1, x2, y2 = map(int, box)
                        # Draw bounding box on the main image
                        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Optionally, display the class name
                        cv.putText(img, 'Car', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the video frame
        cv.imshow('Video Stream', img)

        # Exit on 'q' key press
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cv.destroyAllWindows()
