import cv2

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