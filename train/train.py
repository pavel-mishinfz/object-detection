import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO model for satellite ship detection")
    parser.add_argument("--data", default="config/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Device: '' for auto, '0' for GPU 0, 'cpu'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs", help="Output directory")
    parser.add_argument("--name", default="v1", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing experiment")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(__file__).parent / args.data

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
    )

    print(f"\nBest weights saved to: {model.trainer.best}")


if __name__ == "__main__":
    main()
