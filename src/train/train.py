from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='yolo11n.pt', help='Pretrained model path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    # 載入模型
    model = YOLO(args.model)

    # 開始訓練
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

if __name__ == '__main__':
    main()
