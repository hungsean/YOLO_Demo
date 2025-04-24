from utils.extract_frames import extract_frames
from utils.infer_images import infer_images
from recognize.yolo_results_to_video import results_to_video

def recognize_video(model, video_path: str, output_path: str, fps_extract: int = 1, fps_output: int = 24):
    frames = extract_frames(video_path, fps_extract)
    results = infer_images(model, frames)
    results_to_video(results, output_path, fps=fps_output)
