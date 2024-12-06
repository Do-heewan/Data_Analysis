# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture(0)

model = YOLO('yolov8n-pose.pt')

while cap.isOpened():
  success, image = cap.read()
  image = cv2.flip(image, 1)
  if not success:
    continue

  results = model(image)
  cv2.imshow('img', results[0].plot())

  if cv2.waitKey(1) == ord('q'):
    break
cap.release()
