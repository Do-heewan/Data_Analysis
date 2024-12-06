from ultralytics import YOLO, solutions
import cv2

c_width = 1280
c_height = 720
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_height)

model = YOLO('yolov8n.pt')

assert cap.isOpened(), "Error reading video file"

# Define line points
line_points = [(int(c_width/2), 0), (int(c_width/2), c_height)]

counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    im0 = cv2.flip(im0, 1)
    if not success:
        continue

    tracks = model.track(im0, persist=True, show=False, classes=[0])
    im0 = counter.start_counting(im0, tracks)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()



