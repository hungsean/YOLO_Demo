# tests/test_video_extractor.py

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.utils.extract_frames import extract_frames

def generate_fake_frames(total_frames, frame_shape=(100, 100, 3)):
    """產生假影格陣列"""
    return [np.ones(frame_shape, dtype=np.uint8) * i for i in range(total_frames)]

@patch("cv2.VideoCapture")
def test_extract_frames(mock_VideoCapture):
    # 模擬影片設定
    fake_fps = 10
    extract_fps = 2
    total_frames = 10

    # 建立假 VideoCapture 行為
    mock_cap = MagicMock()
    fake_frames = generate_fake_frames(total_frames)
    
    # read() 模擬：每次呼叫回傳 (True, frame)，最後回傳 (False, None)
    mock_cap.read.side_effect = [(True, f) for f in fake_frames] + [(False, None)]
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = fake_fps  # get(cv2.CAP_PROP_FPS)
    
    mock_VideoCapture.return_value = mock_cap

    # 呼叫函式
    result = extract_frames("dummy.mp4", fps_extract=extract_fps)

    # 驗證結果：應該每 5 張取 1 張 → 0, 5
    assert len(result) == 2
    assert (result[0] == fake_frames[0]).all()
    assert (result[1] == fake_frames[5]).all()
