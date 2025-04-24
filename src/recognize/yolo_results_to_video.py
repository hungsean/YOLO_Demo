import cv2
import numpy as np
from ultralytics.engine.results import Results

def results_to_video(results: list[Results], output_path: str, fps: int = 24) -> None:
    if not results:
        raise ValueError("Empty results list provided.")

    # 假設所有結果影像大小一樣
    frame_shape = results[0].orig_shape
    height, width = frame_shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for result in results:
        # 取得繪製好的圖像
        rendered = result.plot()  # 回傳 np.ndarray 格式
        video_writer.write(rendered)

    video_writer.release()
    print(f"Video saved to {output_path}")
