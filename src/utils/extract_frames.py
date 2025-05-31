import cv2
import os
import argparse
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

def extract_frames_by_fps(video_path: str, fps_extract: int) -> list:
    """
    依照使用者指定的 fps_extract 值，從影片中擷取等間隔的幀。
    若 fps_extract <= 0，則擷取所有幀。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    # 當 fps_extract <= 0 時，就擷取每一幀；否則把原始 FPS 分段
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

def process_batch_similarity(batch, threshold):
    """
    批次內使用 SSIM 計算相鄰幀之相似度，若低於 threshold 才保留此幀。
    """
    kept = []
    prev_gray = None

    for frame in batch:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            kept.append(frame)
            prev_gray = gray
            continue

        similarity, _ = ssim(prev_gray, gray, full=True)
        #print(f"SSIM: {similarity:.4f}")  # 假如想要看每次的相似值可以解開註解
        if similarity < threshold:
            kept.append(frame)
            prev_gray = gray

    return kept

def extract_frames_by_similarity(video_path: str, threshold: float = 0.95, batch_size: int = 30) -> list:
    """
    將影片依 batch_size 拆分成多個 batch，並用多線程同時計算 SSIM，篩選出相似度低於 threshold 的幀。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    all_batches = []
    current_batch = []

    # 先把整部影片按 batch_size 拆成多個 batch
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_batch.append(frame.copy())

        if len(current_batch) == batch_size:
            all_batches.append(current_batch)
            current_batch = []

    # 剩下不滿 batch_size 的部分也算一個 batch
    if current_batch:
        all_batches.append(current_batch)

    cap.release()

    frames = []
    # 使用 ThreadPoolExecutor 同步跑多個 batch 的 SSIM 計算
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda b: process_batch_similarity(b, threshold), all_batches))

    # 將各 batch 過濾後的幀合併
    for batch_result in results:
        frames.extend(batch_result)

    return frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video based on mode (similarity or fps)")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save extracted images")

    # 新增 mode 參數，可選 'similarity' 或 'fps'
    parser.add_argument(
        "-m", "--mode", choices=["similarity", "fps"], default="similarity",
        help="Mode for frame extraction: 'similarity' 為相似度抽幀；'fps' 為固定 FPS 抽幀 (default: similarity)"
    )

    # 以下參數根據不同模式作彈性設定
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5,
        help="當 mode 為 similarity 時使用的 SSIM 相似度門檻 (0～1)，數值越低就是越敏感地抓取變化 (default: 0.5)"
    )
    parser.add_argument(
        "-f", "--fps", type=int, default=1,
        help="當 mode 為 fps 時，要抽幾 FPS；若 <= 0 則代表擷取所有幀 (default: 1)"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=30,
        help="當 mode 為 similarity 時，用於拆分批次的大小 (default: 30)"
    )

    args = parser.parse_args()

    # 確保輸出資料夾存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 根據 mode 來決定採用哪一種抽幀方式
    if args.mode == "similarity":
        print("使用相似度抽幀模式，門檻值 (threshold)：", args.threshold)
        frames = extract_frames_by_similarity(
            video_path=args.input_file,
            threshold=args.threshold,
            batch_size=args.batch_size
        )
    else:  # args.mode == "fps"
        print("使用 FPS 抽幀模式，目標 FPS：", args.fps)
        frames = extract_frames_by_fps(
            video_path=args.input_file,
            fps_extract=args.fps
        )

    # 把擷取到的幀依序存成 jpg
    for i, frame in enumerate(frames):
        output_path = os.path.join(args.output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"已擷取到 {len(frames)} 張影格並存放在 '{args.output_dir}'")

if __name__ == "__main__":
    main()
