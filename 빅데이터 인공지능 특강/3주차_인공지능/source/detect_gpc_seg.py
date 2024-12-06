# Load YOLOv8n, train it on COCO128 for 3 epochs and predict an image with it
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('road_sample.mp4')

model = YOLO('gpc_seg.pt')

while cap.isOpened():
  success, image = cap.read()
  if not success:
    continue

  results = model(image, conf=0.5)
  cv2.imshow('img', results[0].plot())

  if cv2.waitKey(1) == ord('q'):
    break
cap.release()



