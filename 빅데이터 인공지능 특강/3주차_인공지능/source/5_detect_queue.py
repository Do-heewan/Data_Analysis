import cv2
from ultralytics import YOLO, solutions

c_width = 1280
c_height = 720
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)

model = YOLO("yolov8n.pt")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

queue_region = [(20, 20), (int(c_width/2), 20), (int(c_width/2), c_height-20), (20, c_height-20)]

queue = solutions.QueueManager(
    classes_names=model.names,
    reg_pts=queue_region,
    line_thickness=3,
    fontsize=1.0,
    region_color=(255, 144, 31),
)

while cap.isOpened():
    success, im0 = cap.read()
    im0 = cv2.flip(im0, 1)

    if success:
        tracks = model.track(im0, show=False, persist=True, verbose=False, classes=[0])
        out = queue.process_queue(im0, tracks)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    print("Video frame is empty or video processing has been successfully completed.")
    break

cap.release()
cv2.destroyAllWindows()