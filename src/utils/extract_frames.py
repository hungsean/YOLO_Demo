import cv2
import os
import argparse
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

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

def process_batch(batch, threshold):
    kept = []
    prev_gray = None

    for frame in batch:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            kept.append(frame)
            prev_gray = gray
            continue

        similarity, _ = ssim(prev_gray, gray, full=True)
        print(similarity)
        if similarity < threshold:
            kept.append(frame)
            prev_gray = gray

    return kept

def extract_frames_multithread(video_path: str, threshold: float = 0.95, batch_size: int = 30) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    all_batches = []
    current_batch = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_batch.append(frame.copy())

        if len(current_batch) == batch_size:
            all_batches.append(current_batch)
            current_batch = []

    if current_batch:
        all_batches.append(current_batch)

    cap.release()

    frames = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda b: process_batch(b, threshold), all_batches))

    for batch_result in results:
        frames.extend(batch_result)

    return frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video based on visual difference")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("output_dir", help="Directory to save extracted images")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="SSIM similarity threshold (0-1), lower means more sensitive to changes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frames = extract_frames_multithread(args.input_file, args.threshold)
    for i, frame in enumerate(frames):
        output_path = os.path.join(args.output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"Extracted {len(frames)} frames to '{args.output_dir}'")

if __name__ == "__main__":
    main()
