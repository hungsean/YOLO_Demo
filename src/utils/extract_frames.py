import cv2
import os
import argparse

def extract_frames(video_path: str, fps_extract: int) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # 若 fps_extract <= 0，則每一幀都擷取
    extract_interval = 1 if fps_extract <= 0 else max(1, int(original_fps / fps_extract))
    frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % extract_interval == 0:
            frames.append(frame.copy())
        frame_idx += 1

    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("fps", type=int, help="Number of frames per second to extract")
    parser.add_argument("output_dir", help="Directory to save extracted images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frames = extract_frames(args.input_file, args.fps)
    for i, frame in enumerate(frames):
        output_path = os.path.join(args.output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"Extracted {len(frames)} frames to '{args.output_dir}'")

if __name__ == "__main__":
    main()
