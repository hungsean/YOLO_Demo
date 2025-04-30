from ultralytics import YOLO
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('-m', '--model', type=str, help='Pretrained model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    # 載入模型
    model = YOLO(args.model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # 開始訓練
    model.train(
        data=args.data, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        batch=args.batch, 
        # patience=20,
        name=f"yolo_{timestamp}"
        )

if __name__ == '__main__':
    main()
