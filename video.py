import cv2
import av

container = av.open('rtsp://admin:@192.168.1.10:554/stream1')

for packet in container.demux(video=0):
    for frame in packet.decode():
        img = frame.to_ndarray(format='bgr24')
        cv2.imshow('Video Stream', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()