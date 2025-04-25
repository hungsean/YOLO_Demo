from ultralytics import YOLO
from recognize.recognize_video import recognize_video

model = YOLO("runs\\detect\\train3\\weights\\best.pt")
recognize_video(model, "datasets\\test\\VID20250425184101.mp4", "output\\test-4.mp4", fps_extract=0)
