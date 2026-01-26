import cv2
from ultralytics import YOLO
import numpy as np

############ First check if MPS is available! MPS is like CUDA but for Macbooks with M1 or M2 chips ############
import torch
print(torch.backends.mps.is_available())

############ next part is the actual code for object detection ############
cap = cv2.VideoCapture("manwalking.mp4")

model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device ="mps" if torch.backends.mps.is_available() else "cpu", conf = 0.4, iou = 0.5)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype=int)
    classes = np.array(result.boxes.cls.cpu(), dtype=int)

    for cls, bbox in zip(classes, bboxes):
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 225), 2)
        cv2.putText(frame, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 2)

    print("Bounding Boxes:", bboxes)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

cap.release()
cv2.destroyAllWindows()
