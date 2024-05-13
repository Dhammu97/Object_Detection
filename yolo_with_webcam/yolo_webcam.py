# 48:08 minutes completed

from ultralytics import YOLO
import cv2
import cvzone
import math

# This is for a webcam-----
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# This is for a downloaded video -----
# cap = cv2.VideoCapture("video path")


model = YOLO("../yolo_weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane",
              "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
              "stop sign", "parking meter", "bench", "bird", "cat", "dog",
              "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sport ball", "kits", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "brocoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant",
              "bed", "dining table", "toilet", "tv monitor", "laptop",
              "mouse", "remote", "keyboard", "cell phone", "micro wave",
              "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
              "scissors", "teddy bear", "hair drier", "toothbrush"]

while True:
    success, img = cap.read()
    result = model(img, stream=True)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence

            conf = math.ceil((box.conf[0] * 100)) / 100
            cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)))
            # Class Name

            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)))

    cv2.imshow("Image", img)
    cv2.waitKey(1)
