from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os

# ---- Config ----
VIDEO_PATH = 'tacticam.mp4'    #change path for broadcast.mp4
MODEL_PATH = 'best.pt'
OUTPUT_VIDEO = 'tracked_videos/tacticam_tracked.mp4' #change it here too as tracked_videos/broadcast_tracked.mp4
os.makedirs('tracked_videos', exist_ok=True)

# ---- Load Models ----
yolo_model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)  # Keeps a player ID alive for ~30 frames without seeing them again

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# ---- Process Video ----
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo_model.predict(source=frame, conf=0.5)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        bbox = (x1, y1, x2 - x1, y2 - y1)  # Format as tuple
        detections.append((bbox, conf, 0))  # Format: (box, conf, class)

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Finished tracking, saved to {OUTPUT_VIDEO}")
