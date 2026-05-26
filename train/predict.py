import argparse
import cv2
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a directory of images")
    parser.add_argument("--weights", default="runs/v1/weights/best.pt", help="Path to model weights")
    parser.add_argument("--source", required=True, help="Directory with input images")
    parser.add_argument("--output", default="predict", help="Directory for annotated output images")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="Device: '' for auto, '0' for GPU 0, 'cpu'")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width in pixels")
    return parser.parse_args()


def main():
    args = parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise ValueError(f"--source must be a directory, got: {source}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    total_images = 0
    total_detections = 0

    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=False,
        stream=True,
    )

    for result in results:
        annotated = result.plot(line_width=args.line_width)
        out_path = output_dir / Path(result.path).name
        cv2.imwrite(str(out_path), annotated)

        total_images += 1
        total_detections += len(result.boxes)

    print(f"\nDone: {total_images} images, {total_detections} detections")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
