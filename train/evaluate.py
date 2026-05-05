import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO model")
    parser.add_argument("--weights", default="runs/v1/weights/best.pt", help="Path to model weights")
    parser.add_argument("--data", default="config/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="Device: '' for auto, '0' for GPU 0, 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(__file__).parent / args.data

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    mp = metrics.box.mp
    mr = metrics.box.mr
    f1 = 2 * mp * mr / (mp + mr + 1e-9)

    print("\n=== Evaluation Results ===")
    print(f"Split           : {args.split}")
    print(f"Weights         : {args.weights}")
    print(f"mAP@0.5         : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95    : {metrics.box.map:.4f}")
    print(f"Precision       : {mp:.4f}")
    print(f"Recall          : {mr:.4f}")
    print(f"F1              : {f1:.4f}")


if __name__ == "__main__":
    main()
