import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

class VideoObjectDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def infer_frame(self, frame):
        """
        將單張 frame 丟進 YOLO 模型推論，回傳 numpy 格式的 boxes。
        每個 box 為 [x1, y1, x2, y2, conf, cls]
        """
        results = self.model(frame)  # 傳入 BGR numpy 圖像，Ultralytics 會自動處理轉 RGB + resize
        boxes = results[0].boxes.data.cpu().numpy()  # shape: (N, 6)
        return boxes

    def draw_predictions(self, frame, predictions):
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = map(int, pred[:6])  # 視你的模型輸出格式調整
            label = f"{int(cls)}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def process_video(self, input_path: str, output_path: str, fps_extract: int = 5):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_idx = 0
        extract_interval = int(fps / fps_extract)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % extract_interval == 0:
                preds = self.infer_frame(frame)
                frame = self.draw_predictions(frame, preds)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Saved output video to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run object detection on a video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--fps", type=int, default=5, help="How many frames per second to extract")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu)")

    args = parser.parse_args()

    detector = VideoObjectDetector(args.model)
    detector.process_video(args.input, args.output, fps_extract=args.fps)