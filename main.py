from ultralytics import YOLO

# 載入預訓練好的 YOLOv11n 模型（可換成 yolov11s.pt, yolov11m.pt 等）
model = YOLO("models\\yolo11x.pt")
model.to("cuda")

# 對一張圖片進行物件偵測
results = model("datasets\\test\\IMG20250424193338.jpg")  # 替換成你實際的圖片路徑

print(results[0])
# 顯示結果（自帶繪製邊框與 label）
results[0].show()

