import cv2

def create_tracker():
    return cv2.TrackerCSRT_create()

def initialize_trackers(frame, detections):
    trackers = []
    for det in detections:
        tracker = create_tracker()
        tracker.init(frame, tuple(det))
        trackers.append(tracker)
    return trackers

def update_trackers(frame, trackers):
    new_detections = []
    for tracker in trackers:
        success, box = tracker.update(frame)
        if success:
            new_detections.append(box)
    return new_detections

# Usage in main loop
trackers = initialize_trackers(first_frame, initial_detections)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    tracked_players = update_trackers(frame, trackers)
    
    # Draw tracked players on frame
    for player in tracked_players:
        x, y, w, h = [int(v) for v in player]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Tracked Players', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()