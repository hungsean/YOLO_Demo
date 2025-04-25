from ultralytics import YOLO
from recognize.recognize_video import recognize_video
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize objects in a video using a YOLO model.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the trained model or model identifier.")
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to save the output video.")
    parser.add_argument("--fps_extract", type=int, default=0, help="FPS rate for extracting frames.")
    parser.add_argument("--fps_output", type=int, default=30, help="FPS rate for output video.")

    args = parser.parse_args()

    # 假設你的model是直接load進來的（如果需要自定義可以再調整這段）
    model = YOLO(args.model)

    recognize_video(
        model,
        args.video_path,
        args.output_path,
        fps_extract=args.fps_extract,
        fps_output=args.fps_output
    )
