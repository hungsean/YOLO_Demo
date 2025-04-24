import pytest
import src.recognize.recognize_video as recognize_video_module

def test_recognize_video_flow(monkeypatch, tmp_path):
    calls = {}

    def fake_extract_frames(path, fps):
        calls['frames'] = (path, fps)
        return ['frameA', 'frameB']

    def fake_infer_images(model, frames):
        calls['images'] = (model, frames)
        return ['res1', 'res2']

    def fake_results_to_video(results, out_path, fps):
        calls['video'] = (results, out_path, fps)

    # 替換 recognize_video 模組中的子函式
    monkeypatch.setattr(recognize_video_module, 'extract_frames', fake_extract_frames)
    monkeypatch.setattr(recognize_video_module, 'infer_images', fake_infer_images)
    monkeypatch.setattr(recognize_video_module, 'results_to_video', fake_results_to_video)


    dummy_model = object()
    video_path = "in.mp4"
    output_path = str(tmp_path / "out.mp4")
    recognize_video_module.recognize_video(dummy_model, video_path, output_path,
                                    fps_extract=3, fps_output=12)

    assert calls['frames'] == (video_path, 3)
    assert calls['images'] == (dummy_model, ['frameA', 'frameB'])
    assert calls['video'] == (['res1', 'res2'], output_path, 12)
