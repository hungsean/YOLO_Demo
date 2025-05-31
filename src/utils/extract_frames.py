import cv2
import os
import argparse
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

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
        if similarity < threshold:
            kept.append(frame)
            prev_gray = gray

    return kept

def extract_frames_by_similarity(video_path: str, threshold: float = 0.95, batch_size: int = 30, 
                                max_workers: int = None, max_pending_batches: int = None) -> list:
    """
    輸入影片路徑，基於 SSIM 閾值抽幀。這個版本會：
      1. 一邊讀影片，一邊把每 batch_size 幀「丟給 ThreadPoolExecutor 處理」
      2. 限制最多只能同時 pending（等待計算）的 batch 數（用 max_pending_batches 控制）
      3. 每一個 batch 處理完就把結果加入 frames，並釋放這個 batch 的記憶
    
    這樣在保持平行運算加速的同時，也避免一次把所有 batch 都塞進記憶體。
    
    參數：
      - video_path：影片檔路徑
      - threshold：SSIM 門檻值（0~1）
      - batch_size：每批要讀幾幀（預設 30）
      - max_workers：ThreadPoolExecutor 的最大 worker 數（None 代表預設 CPU 數）
      - max_pending_batches：同時間最多允許多少還沒算完的 batch（預設同 max_workers * 2）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"無法開啟影片：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    current_batch = []
    futures = []
    
    # 如果外面沒特別指定，max_workers 就用系統預設 (CPU 數量)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    # 如果沒給 max_pending_batches，就預設讓待處理的 batch 最多是 worker 數量的兩倍
    if max_pending_batches is None:
        max_pending_batches = (executor._max_workers or 1) * 2

    # 進度條：按照「影片總幀數」跑
    pbar = tqdm(total=total_frames, desc="Similarity 抽幀總進度", unit="幀")

    # -----------------------------------
    #  一邊讀影片、一邊丟 batch 去 executor 處理
    # -----------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            # 影片讀完，還要把最後沒滿 batch_size 的那一批也 submit
            if current_batch:
                # 複製出一個新 list 以免後面被改動到
                batch_copy = current_batch.copy()
                future = executor.submit(process_batch_similarity, batch_copy, threshold)
                futures.append(future)
                current_batch.clear()
            break

        current_batch.append(frame.copy())
        pbar.update(1)

        if len(current_batch) == batch_size:
            # 拿到滿 batch_size 的 batch，立刻 submit 給 executor
            batch_copy = current_batch.copy()
            future = executor.submit(process_batch_similarity, batch_copy, threshold)
            futures.append(future)
            # 清空 current_batch，繼續下一批
            current_batch.clear()

        # 如果同時 pending 未完成的 futures 超過上限，就先「等待至少一個完成」再繼續新增
        if len(futures) >= max_pending_batches:
            done, not_done = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                kept_list = fut.result()   # 拿到這個 batch 的處理結果（就是過濾後的 frames）
                frames.extend(kept_list)
            # 把已完成的 futures 從列表中移除
            futures = list(not_done)

    # 讀完影片後，還有可能剩下「還沒完成」的 futures，要等它們都做完
    for fut in as_completed(futures):
        kept_list = fut.result()
        frames.extend(kept_list)

    # 關閉進度條、釋放資源
    pbar.close()
    cap.release()
    executor.shutdown(wait=True)

    return frames

def extract_frames_by_fps(video_path: str, fps_extract: int) -> list:
    """
    固定 FPS 抽幀（同前面一樣，只做一邊讀一邊記錄並更新進度條）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    extract_interval = 1 if fps_extract <= 0 else max(1, int(original_fps / fps_extract))

    frames = []
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="FPS 抽幀進度", unit="幀")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pbar.update(1)
        if frame_idx % extract_interval == 0:
            frames.append(frame.copy())
        frame_idx += 1

    pbar.close()
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video based on mode (similarity or fps)")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save extracted images")

    parser.add_argument(
        "-m", "--mode", choices=["similarity", "fps"], default="similarity",
        help="Mode for frame extraction: 'similarity' 為相似度抽幀；'fps' 為固定 FPS 抽幀 (default: similarity)"
    )

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
    parser.add_argument(
        "--max_workers", type=int, default=None,
        help="ThreadPoolExecutor 的最大 worker 數 (default: CPU 核心數)"
    )
    parser.add_argument(
        "--max_pending", type=int, default=None,
        help="同時間允許最多 pending batch 的數量 (default: max_workers * 2)"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "similarity":
        print("使用相似度抽幀模式，門檻值 (threshold)：", args.threshold)
        frames = extract_frames_by_similarity(
            video_path=args.input_file,
            threshold=args.threshold,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            max_pending_batches=args.max_pending
        )
    else:
        print("使用 FPS 抽幀模式，目標 FPS：", args.fps)
        frames = extract_frames_by_fps(
            video_path=args.input_file,
            fps_extract=args.fps
        )

    # 把擷取到的幀依序寫成 jpg
    for i, frame in enumerate(frames):
        output_path = os.path.join(args.output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"已擷取到 {len(frames)} 張影格並存放在 '{args.output_dir}'")

if __name__ == "__main__":
    main()
