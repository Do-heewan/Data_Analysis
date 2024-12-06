import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"

# Init heatmap
heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=True,
    heatmap_alpha=1,
    shape="circle",
    classes_names=model.names,
)

while cap.isOpened():
    success, im0 = cap.read()
    im0 = cv2.flip(im0, 1)
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=True, classes=[0])

    im0 = cv2.flip(im0, 1)
    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    if cv2.waitKey(1) == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()