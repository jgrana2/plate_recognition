import cv2 as cv
import av
import numpy as np
from ultralytics import YOLO
import time
import logging
from image_api import print_license_plate
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('car_detection.log'),
        logging.StreamHandler()  # This will print to console as well
    ]
)
logger = logging.getLogger(__name__)

def iou(box1, box2):
    """ Calculate the Intersection over Union (IoU) between two bounding boxes. """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou_score = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
    logger.debug(f"IoU calculation: score={iou_score:.4f}")
    return iou_score

# Set the logging level for 'ultralytics' to WARNING to suppress INFO messages
logging.getLogger('ultralytics').setLevel(logging.WARNING)

logger.info("Starting car detection system...")

# Load the YOLOv8 model
logger.info("Loading YOLO model...")
np_model = YOLO('yolo11n.pt')
logger.info("YOLO model loaded successfully")

# Get the class names
class_names = np_model.names
logger.info(f"Model loaded with {len(class_names)} classes")

# Find the class IDs for 'car'
car_class_ids = [cls_id for cls_id, name in class_names.items() if name.lower() == 'car']
logger.info(f"Car class IDs identified: {car_class_ids}")

if not car_class_ids:
    logger.error("Class 'car' not found in the model's class names.")
    exit()

# Load the video stream
logger.info("Connecting to RTSP stream...")
try:
    container = av.open('rtsp://admin:@192.168.1.10:554/stream1')
    logger.info("Successfully connected to RTSP stream")
except Exception as e:
    logger.error(f"Failed to connect to RTSP stream: {str(e)}")
    exit()

# Set the detection interval in seconds
detection_interval = 1.0
logger.info(f"Detection interval set to {detection_interval} seconds")

# Initialize tracking variables
last_detection_time = time.time()
last_detections = None
best_confidence = 0.0
best_car_img_masked = None
detected_boxes = []
frame_count = 0
cars_detected = 0

logger.info("Starting main processing loop...")

for packet in container.demux(video=0):
    for frame in packet.decode():
        frame_count += 1
        img = frame.to_ndarray(format='bgr24')
        
        logger.debug(f"Processing frame {frame_count}")
        
        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            logger.debug("Running detection on current frame")
            
            # Run YOLO detection
            results = np_model(img)
            last_detections = results
            last_detection_time = current_time

            car_detected_in_frame = False
            max_confidence_in_frame = 0.0
            current_car_id = None
            current_detected_boxes = []

            if last_detections is not None:
                for result in last_detections:
                    boxes = result.boxes.xyxy
                    class_ids = result.boxes.cls
                    confidences = result.boxes.conf
                    
                    num_detections = len(boxes)
                    logger.debug(f"Found {num_detections} objects in frame")
                    
                    for box, cls_id, conf in zip(boxes, class_ids, confidences):
                        if int(cls_id) in car_class_ids:
                            logger.debug(f"Car detected with confidence: {conf:.4f}")
                            current_detected_boxes.append(box)
                            car_detected_in_frame = True

                            # Check if this is a new car
                            is_new_car = True
                            for prev_box in detected_boxes:
                                if iou(prev_box, box) > 0.5:
                                    is_new_car = False
                                    logger.debug("Matched with previously detected car")
                                    break
                            
                            if is_new_car:
                                cars_detected += 1
                                logger.info(f"New car detected! (Total cars: {cars_detected})")
                                logger.info(f"Detection confidence: {conf:.4f}")
                                best_confidence = 0.0
                                best_car_img_masked = None

                            if is_new_car and conf > max_confidence_in_frame:
                                max_confidence_in_frame = conf
                                x1, y1, x2, y2 = map(int, box)
                                
                                logger.debug(f"Processing new car image at coordinates: ({x1}, {y1}, {x2}, {y2})")
                                
                                # Extract the car image
                                car_img = img[y1:y2, x1:x2].copy()
                                
                                # Create mask dimensions
                                height, width = car_img.shape[:2]
                                logger.debug(f"Extracted car image dimensions: {width}x{height}")
                                
                                # Create a roundbox mask
                                radius = int(min(width, height) / 10)
                                mask = np.zeros((height, width), dtype=np.uint8)
                                cv.rectangle(mask, (0, 0), (width, height), 255, -1)
                                cv.circle(mask, (radius, radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, radius), radius, 0, -1)
                                cv.circle(mask, (radius, height - radius), radius, 0, -1)
                                cv.circle(mask, (width - radius, height - radius), radius, 0, -1)

                                best_car_img_masked = car_img

                if car_detected_in_frame:
                    if max_confidence_in_frame > best_confidence:
                        best_confidence = max_confidence_in_frame
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f'car_{timestamp}.png'
                        logger.info(f"Saving car image with confidence {best_confidence:.4f} to {filename}")
                
                        if best_car_img_masked is not None and best_car_img_masked.size > 0:
                            cv.imwrite(filename, best_car_img_masked)
                            logger.info("Image saved successfully")
                            
                            logger.info("Starting license plate processing thread")
                            threading.Thread(target=print_license_plate).start()
                        else:
                            logger.error("Failed to save image: Empty or invalid image data")

            if current_detected_boxes is not None:
                detected_boxes = current_detected_boxes
                logger.debug(f"Updated tracking boxes: {len(detected_boxes)} cars")

        # Draw detections
        if last_detections is not None:
            for result in last_detections:
                boxes = result.boxes.xyxy
                class_ids = result.boxes.cls
                for box, cls_id in zip(boxes, class_ids):
                    if int(cls_id) in car_class_ids:
                        x1, y1, x2, y2 = map(int, box)
                        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv.putText(img, 'Car', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the frame
        cv.imshow('Video Stream', img)

        if cv.waitKey(10) & 0xFF == ord('q'):
            logger.info("User requested exit")
            break

logger.info(f"Processing complete. Total frames processed: {frame_count}")
logger.info(f"Total cars detected: {cars_detected}")
logger.info("Cleaning up and exiting...")

cv.destroyAllWindows()