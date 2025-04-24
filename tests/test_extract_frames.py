import pytest
import numpy as np
import cv2
from src.utils.extract_frames import extract_frames

class DummyCapture:
    def __init__(self, frames, fps, fail_open=False):
        self.frames = frames
        self.fps = fps
        self.fail_open = fail_open
        self.idx = 0
    def isOpened(self):
        return not self.fail_open
    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return self.fps
        return 0
    def read(self):
        if self.idx >= len(self.frames):
            return False, None
        frame = self.frames[self.idx]
        self.idx += 1
        return True, frame
    def release(self):
        pass

@pytest.fixture(autouse=True)
def patch_videocapture(monkeypatch):
    def _factory(path):
        # 模擬一個有 5 幀的影像，每幀都是 2x2 全 1
        dummy_frames = [np.ones((2,2,3), dtype=np.uint8) for _ in range(5)]
        return DummyCapture(dummy_frames, fps=5.0, fail_open=False)
    monkeypatch.setattr(cv2, 'VideoCapture', _factory)

def test_open_fail(monkeypatch):
    # 當無法開啟影片時，應該丟出 IOError
    def bad_factory(path):
        return DummyCapture([], fps=0, fail_open=True)
    monkeypatch.setattr(cv2, 'VideoCapture', bad_factory)
    with pytest.raises(IOError):
        extract_frames("nonexistent.mp4", fps_extract=1)

def test_extract_all_frames_when_fps_extract_le_zero():
    # fps_extract <= 0 時，應該回傳所有幀
    frames = extract_frames("dummy.mp4", fps_extract=0)
    assert len(frames) == 5

def test_extract_by_interval():
    # 原始 fps=5，fps_extract=2，則 interval = max(1, int(5/2)) = 2
    frames = extract_frames("dummy.mp4", fps_extract=2)
    # 5 幀中每兩幀取一幀，應該取到 3 幀 (idx 0,2,4)
    assert len(frames) == 3
