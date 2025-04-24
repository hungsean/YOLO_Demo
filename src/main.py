from utils.extract_frames import extract_frames
from utils.infer_images import infer_images
import cv2
from ultralytics import YOLO

video_path = "datasets\\test\\VID20250424195257.mp4"
fps_extract = 0

frames = extract_frames(video_path, fps_extract)
print(f"Extracted {len(frames)} frames.")

for i, frame in enumerate(frames[:5]):  # 儲存前 5 張
    cv2.imwrite(f"datasets\\test\\frame_{i:03d}.jpg", frame)
print("Saved first 5 frames to current directory.")

model = YOLO("models/yolo11x.pt")

results = infer_images(model, frames)
print(f"Inferred {len(results)} images.")
# results[10].plot()