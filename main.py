import os
os.chdir(r"")
from ultralytics import YOLO
import cv2

results = {}

coco_model = YOLO('yolov8n.pt')
license_plate_Detection = YOLO('')

vehicles = [2, 3, 5, 7]

cap = cv2.VideoCapture('')

frame_no = -1
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        # use YOLO model on every frame
        results[frame_no] = {}

        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        license_plates = license_plate_Detection(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
                                            
