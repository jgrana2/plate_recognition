import cv2 as cv
import av
import numpy as np
from ultralytics import YOLO
import time
import logging
from image_api import print_license_plate
import threading
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('car_detection.log'),
        logging.StreamHandler()
    ]
)

# Suppress unnecessary logging from libraries
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('av').setLevel(logging.WARNING)

# Create logger for this application
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, model_path='yolo11n.pt'):
        logger.info("Initializing VideoProcessor...")
        
        # Load the YOLO model
        logger.info("Loading YOLO model...")
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        
        # Get class names and car class IDs
        self.class_names = self.model.names
        self.car_class_ids = [cls_id for cls_id, name in self.class_names.items() 
                              if name.lower() == 'car']
        
        if not self.car_class_ids:
            logger.error("Class 'car' not found in model's class names.")
            raise ValueError("Car class not found in model")
            
        logger.info(f"Car class IDs identified: {self.car_class_ids}")
        
        # Processing parameters
        self.detection_interval = 1.0  # seconds
        self.last_detection_time = time.time()
        self.last_detections = None
        self.best_confidence = 0.0
        self.best_car_img_masked = None
        self.detected_boxes = []
        self.frame_count = 0
        self.cars_detected = 0
        
    def iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
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

    def process_frame(self, frame, source_name):
        """Process a single frame for car detection"""
        self.frame_count += 1
        current_time = time.time()
        
        # Initialize car_detections to ensure it's always defined
        car_detections = []
        
        # Run detection at intervals
        if current_time - self.last_detection_time >= self.detection_interval:
            logger.debug(f"Running detection on frame {self.frame_count} from {source_name}")
            
            results = self.model(frame)
            self.last_detections = results
            self.last_detection_time = current_time
            
            current_detected_boxes = []
            
            if self.last_detections is not None:
                for result in self.last_detections:
                    boxes = result.boxes.xyxy
                    class_ids = result.boxes.cls
                    confidences = result.boxes.conf
                    
                    logger.debug(f"Found {len(boxes)} objects in frame")
                    
                    for box, cls_id, conf in zip(boxes, class_ids, confidences):
                        if int(cls_id) in self.car_class_ids:
                            logger.debug(f"Car detected with confidence: {conf:.4f}")
                            current_detected_boxes.append(box)
                            car_detections.append({
                                'box': box,
                                'confidence': conf
                            })
            
            if car_detections:
                # Identify the largest car based on bounding box area
                largest_car = max(
                    car_detections, 
                    key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1])
                )
                largest_box = largest_car['box']
                largest_conf = largest_car['confidence']
                
                # Check if it's a new car
                is_new_car = True
                for prev_box in self.detected_boxes:
                    if self.iou(prev_box, largest_box) > 0.5:
                        is_new_car = False
                        break
                
                if is_new_car:
                    self.cars_detected += 1
                    logger.info(f"New car detected in {source_name}! (Total: {self.cars_detected})")
                    logger.info(f"Detection confidence: {largest_conf:.4f}")
                    self.best_confidence = 0.0
                    self.best_car_img_masked = None
                    
                    x1, y1, x2, y2 = map(int, largest_box)
                    car_img = frame[y1:y2, x1:x2].copy()
                    
                    if car_img is not None and car_img.size > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f'cars/car_{source_name}_{timestamp}.png'
                        os.makedirs('cars', exist_ok=True)
                        cv.imwrite(filename, car_img)
                        logger.info(f"Saved car image to {filename}")
                        
                        # Start license plate processing in a separate thread with the filename
                        logger.info(f"Starting license plate processing for {filename}")
                        threading.Thread(
                            target=print_license_plate,
                            args=(filename,),
                            daemon=True  # Daemonize thread to exit with main program
                        ).start()
                
                # Update detected boxes with only the largest car
                self.detected_boxes = [largest_box]
            else:
                # No cars detected in this frame
                self.detected_boxes = []
        
        # Draw bounding box around the largest detected car only
        if self.last_detections is not None and self.detected_boxes:
            largest_box = self.detected_boxes[0]
            
            x1, y1, x2, y2 = map(int, largest_box)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                frame, 
                'Car', 
                (x1, y1 - 10), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
        
        return frame

    def process_rtsp_stream(self, rtsp_url):
        """Process RTSP stream"""
        logger.info(f"Connecting to RTSP stream: {rtsp_url}")
        try:
            container = av.open(rtsp_url)
            logger.info("Successfully connected to RTSP stream")
            
            for packet in container.demux(video=0):
                for frame in packet.decode():
                    img = frame.to_ndarray(format='bgr24')
                    processed_frame = self.process_frame(img, "rtsp")
                    
                    cv.imshow('RTSP Stream', processed_frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested exit from RTSP stream")
                        return
                    
        except Exception as e:
            logger.error(f"Error processing RTSP stream: {str(e)}")
        finally:
            cv.destroyAllWindows()

    def process_video_file(self, video_path):
        """Process a single video file"""
        logger.info(f"Processing video file: {video_path}")
        try:
            cap = cv.VideoCapture(str(video_path))
            filename = Path(video_path).stem
            
            # Get video properties
            fps = cap.get(cv.CAP_PROP_FPS)
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video properties - FPS: {fps}, Total frames: {total_frames}")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame, filename)
                
                # Calculate progress percentage
                progress = (self.frame_count / total_frames) * 100
                logger.debug(f"Processing progress: {progress:.1f}%")
                
                cv.imshow(f'Video: {filename}', processed_frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    logger.info(f"User requested exit from video: {filename}")
                    break
                    
        except Exception as e:
            logger.error(f"Error processing video file {video_path}: {str(e)}")
        finally:
            cap.release()
            cv.destroyAllWindows()

    def process_video_folder(self, folder_path):
        """Process all video files in a folder"""
        logger.info(f"Processing videos from folder: {folder_path}")
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.m4v', '.mp3')
        
        try:
            video_files = [f for f in Path(folder_path).glob('*') 
                          if f.suffix.lower() in video_extensions]
            
            if not video_files:
                logger.warning(f"No video files found in {folder_path}")
                return
                
            logger.info(f"Found {len(video_files)} video files")
            
            for video_file in video_files:
                logger.info(f"Starting processing of {video_file}")
                self.process_video_file(video_file)
                logger.info(f"Completed processing of {video_file}")
                
        except Exception as e:
            logger.error(f"Error processing video folder: {str(e)}")

def main():
    """Main function to run the video processor"""
    processor = VideoProcessor()
    
    # Process video files first
    video_folder = "videos/"
    logger.info("Starting video folder processing...")
    processor.process_video_folder(video_folder)
    logger.info("Completed video folder processing")
    
    # Then process RTSP stream
    rtsp_url = 'rtsp://admin:@192.168.1.10:554/stream1'
    logger.info("Starting RTSP stream processing...")
    processor.process_rtsp_stream(rtsp_url)
    logger.info("Completed RTSP stream processing")
    
    # Print final statistics
    logger.info("=== Processing Complete ===")
    logger.info(f"Total frames processed: {processor.frame_count}")
    logger.info(f"Total cars detected: {processor.cars_detected}")

if __name__ == "__main__":
    main()
