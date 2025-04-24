from ultralytics import YOLO
from recognize.recognize_video import recognize_video

model = YOLO("models/yolo11x.pt")
recognize_video(model, "datasets\\test\\VID20250424195257.mp4", "output\\test-3.mp4", fps_extract=0)
