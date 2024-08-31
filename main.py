from track import *
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

capt = cv2.VideoCapture("C:/Users/Luqman/OneDrive/Documents/luqman/Proj/cam/england_epl/2014-2015/source/2_720p.mkv")
# capt = cv2.VideoCapture(0) #forcamera
trackers = []

frame_count = 0
detection_interval = 30

while True:
    ret, frame = capt.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % detection_interval == 0 or frame_count == 1:
        results = model(frame)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        person_boxes = boxes[results[0].boxes.cls.cpu().numpy() == 0]
        
        trackers = initialize_trackers(frame, person_boxes)
    else:
        tracked_players, trackers = update_trackers(frame, trackers)
        
        for player in tracked_players:
            x, y, w, h = player
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('the box', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('0'):
        break
    
capt.release()
cv2.destroyAllWindows()
