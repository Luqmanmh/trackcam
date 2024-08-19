import cv2
from ultralytics import YOLO

def create_tracker():
    return cv2.TrackerCSRT_create()

def initialize_trackers(frame, detections):
    trackers = []
    for det in detections:
        tracker = create_tracker()
        bbox = (int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1]))  # Converting to (x, y, w, h) and ensuring integers
        tracker.init(frame, bbox)
        trackers.append({'tracker': tracker, 'bbox': bbox})
    return trackers

def update_trackers(frame, trackers):
    new_detections = []
    updated_trackers = []
    for tracker_info in trackers:
        tracker = tracker_info['tracker']
        success, box = tracker.update(frame)
        if success:
            box = list(map(int, box))
            new_detections.append(box)
            updated_trackers.append({'tracker': tracker, 'bbox': box})
    return new_detections, updated_trackers

model = YOLO('yolov8n.pt')

capt = cv2.VideoCapture("C:/Users/Luqman/OneDrive/Documents/luqman/Proj/cam/england_epl/2014-2015/source/2_720p.mkv")

trackers = []

frame_count = 0
detection_interval = 30

while True:
    ret, frame = capt.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % detection_interval == 0:
        results = model(frame)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        person_boxes = boxes[results[0].boxes.cls.cpu().numpy() == 0]
        
        trackers = initialize_trackers(frame, person_boxes)
    else:
        tracked_players, trackers = update_trackers(frame, trackers)
        
        for player in tracked_players:
            x, y, w, h = player
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break
    
capt.release()
cv2.destroyAllWindows()
