from ultralytics import YOLO
import cv2
import os

# Change these paths
VIDEO_PATH = 'broadcast.mp4' #change path for tacticam.mp4
MODEL_PATH = 'best.pt'  
# Output folder for detections
os.makedirs('detections', exist_ok=True)

def detect_players(video_path, model_path, output_folder):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5)

        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

        # Save frame with detections
        cv2.imwrite(f"{output_folder}/frame_{frame_id:04d}.jpg", frame)
        frame_id += 1

    cap.release()

if __name__ == "__main__":
    detect_players(VIDEO_PATH, MODEL_PATH, 'detections')
