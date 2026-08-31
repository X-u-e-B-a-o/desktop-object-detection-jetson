from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Boxes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models/cup_bottle_mouse_v5_yolov8s.pt",
    PROJECT_ROOT / "runs/detect/runs/train_cup_bottle_mouse_v5_yolov8s/weights/best.pt",
    PROJECT_ROOT / "runs/detect/runs/train_cup_mouse_v4_yolov8s/weights/best.pt",
    PROJECT_ROOT / "runs/detect/runs/train_cup_mouse_v3_cup_more/weights/best.pt",
    PROJECT_ROOT / "runs/detect/runs/train_cup_mouse_v1/weights/best.pt",
]


def default_model_path() -> str:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return str(path)
    return str(MODEL_CANDIDATES[0])


def parse_args():
    parser = ArgumentParser(description="Run YOLO cup/bottle/mouse detection on real images.")
    parser.add_argument("source", help="Image, video, folder, or camera index, for example test_images/desk4.jpg")
    parser.add_argument(
        "--model",
        default=default_model_path(),
        help="Path to YOLO model weights.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Base confidence threshold before class-specific filtering.")
    parser.add_argument("--cup-conf", type=float, default=0.5, help="Minimum confidence to keep cup detections.")
    parser.add_argument("--bottle-conf", type=float, default=0.5, help="Minimum confidence to keep bottle detections.")
    parser.add_argument("--mouse-conf", type=float, default=0.7, help="Minimum confidence to keep mouse detections.")
    parser.add_argument("--iou", type=float, default=0.3, help="NMS IoU threshold. Lower values remove duplicate boxes sooner.")
    parser.add_argument("--name", default="real_test_filtered", help="Output folder name under runs/detect.")
    parser.add_argument("--show", action="store_true", help="Show prediction window while running.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO txt predictions.")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence values with txt predictions.")
    return parser.parse_args()


def next_output_dir(project: Path, name: str) -> Path:
    output_dir = project / name
    if not output_dir.exists():
        return output_dir

    index = 2
    while True:
        candidate = project / f"{name}-{index}"
        if not candidate.exists():
            return candidate
        index += 1


def keep_class_thresholds(result, cup_conf: float, bottle_conf: float, mouse_conf: float):
    if result.boxes is None or len(result.boxes) == 0:
        return result

    data = result.boxes.data
    conf = data[:, -2]
    cls = data[:, -1].int()
    thresholds = {
        "cup": cup_conf,
        "bottle": bottle_conf,
        "mouse": mouse_conf,
    }
    keep = cls == -1
    for cls_id, name in result.names.items():
        if name in thresholds:
            keep |= (cls == int(cls_id)) & (conf >= thresholds[name])
    result.boxes = Boxes(data[keep], result.orig_shape)
    return result


def main():
    args = parse_args()
    model = YOLO(args.model)
    project = PROJECT_ROOT / "runs/detect"
    save_dir = next_output_dir(project, args.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = save_dir / "labels"

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        save=False,
        verbose=True,
    )

    kept = 0
    for result in results:
        result = keep_class_thresholds(result, args.cup_conf, args.bottle_conf, args.mouse_conf)
        kept += len(result.boxes) if result.boxes is not None else 0

        image_name = Path(result.path).name
        result.save(filename=str(save_dir / image_name))
        if args.save_txt:
            labels_dir.mkdir(parents=True, exist_ok=True)
            result.save_txt(labels_dir / f"{Path(result.path).stem}.txt", save_conf=args.save_conf)

    print(
        f"Filtered results saved to {save_dir} "
        f"(cup_conf={args.cup_conf}, bottle_conf={args.bottle_conf}, "
        f"mouse_conf={args.mouse_conf}, kept_boxes={kept})"
    )


if __name__ == "__main__":
    main()
