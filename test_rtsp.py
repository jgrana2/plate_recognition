import cv2
import time

rtsp_url = "rtsp://admin:@190.29.119.66:5554"
print(f"Testing connection to: {rtsp_url}")

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("ERROR: Cannot open RTSP stream")
    print("Possible issues:")
    print("1. Network connectivity to camera")
    print("2. Incorrect RTSP URL")
    print("3. Camera requires different authentication")
    print("4. Port 5554 blocked by firewall")
else:
    print("SUCCESS: Connected to RTSP stream")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print(f"Frame read successfully: {frame.shape}")
        # Save a test frame
        cv2.imwrite("test_frame.jpg", frame)
        print("Saved test frame as test_frame.jpg")
    else:
        print("WARNING: Connected but cannot read frames")
    
    cap.release()