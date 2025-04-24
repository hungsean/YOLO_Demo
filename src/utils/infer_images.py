from typing import List
from ultralytics.engine.results import Results

def infer_images(model, images: List) -> List[Results]:
    """
    使用模型對多張圖片做推論，回傳每張圖片對應的 Results 物件。
    每個物件包含原圖、框框、分類等資訊，支援 .plot(), .boxes 等屬性。
    """
    results = []
    for img in images:
        res = model(img)       # 推論後是 List[Results]，一張圖就一個元素
        results.append(res[0]) # 取出單一圖片的結果物件
    return results
