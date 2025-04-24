import pytest
from src.utils.infer_images import infer_images

class DummyModel:
    def __init__(self, result_obj):
        self.result_obj = result_obj
        self.calls = []
    def __call__(self, img):
        # 模擬回傳 [Results] 形式
        self.calls.append(img)
        return [self.result_obj]

def test_infer_images_basic():
    dummy = object()
    model = DummyModel(dummy)
    images = ["img1", "img2", "img3"]
    results = infer_images(model, images)
    # 每張圖都應該呼叫一次 model，並回傳同一個 result_obj
    assert results == [dummy, dummy, dummy]
    assert model.calls == images

def test_infer_images_empty_list():
    model = DummyModel("x")
    results = infer_images(model, [])
    assert results == []
    # model 不應該被呼叫
    assert model.calls == []
