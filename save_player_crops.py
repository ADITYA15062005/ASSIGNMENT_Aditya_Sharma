from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os

# ---- Config ----
VIDEO_PATH = 'tacticam.mp4'   # Change to 'broadcast.mp4' for the second video
MODEL_PATH = 'best.pt'
OUTPUT_FOLDER = 'crops/tacticam'  # Change to 'crops/broadcast' for the second video
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

yolo_model = YOLO(MODEL_PATH)
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(source=frame, conf=0.5)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        bbox = (x1, y1, x2 - x1, y2 - y1)  # Tuple format
        detections.append((bbox, conf, 0))

    # ---- DeepSORT Tracking ----
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        # Crop the player from the frame
        player_crop = frame[max(0, t):max(0, b), max(0, l):max(0, r)]

        if player_crop.size == 0:
            continue

        crop_filename = f"{OUTPUT_FOLDER}/player_{track_id}_frame_{frame_id:04d}.jpg"
        cv2.imwrite(crop_filename, player_crop)

        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    frame_id += 1

cap.release()
print(f"Saved crops to {OUTPUT_FOLDER}")
