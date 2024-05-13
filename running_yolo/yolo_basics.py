from ultralytics import YOLO
import cv2

madel = YOLO('../yolo_weights/yolov8l.pt')
results = madel("images/1.png", show=True)
cv2.waitKey(0)
