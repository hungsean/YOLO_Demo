import cv2

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
