from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from shutil import copy2

import torch
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_BOTTLE = 39
COCO_CUP = 41


def parse_args():
    parser = ArgumentParser(description="Split old cup labels into cup/bottle and move mouse to class 2.")
    parser.add_argument("--data", default="data/combined", help="YOLO dataset root.")
    parser.add_argument("--model", default="yolov8s.pt", help="COCO pretrained YOLO model.")
    parser.add_argument("--coco-conf", type=float, default=0.05, help="Low COCO confidence threshold for relabel suggestions.")
    parser.add_argument("--iou", type=float, default=0.25, help="Minimum IoU between dataset label and COCO bottle/cup prediction.")
    parser.add_argument("--backup", default="data/label_backup_before_cup_bottle_split", help="Backup folder for labels and data.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing files.")
    return parser.parse_args()


def yolo_to_xyxy(label, width: int, height: int):
    _, cx, cy, box_w, box_h = label
    x1 = (cx - box_w / 2) * width
    y1 = (cy - box_h / 2) * height
    x2 = (cx + box_w / 2) * width
    y2 = (cy + box_h / 2) * height
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


def box_iou_one_to_many(box, boxes):
    if boxes.numel() == 0:
        return torch.empty(0)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = box_area + boxes_area - inter
    return inter / union.clamp(min=1e-6)


def read_label_file(label_path: Path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    labels = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Expected YOLO box label with 5 columns: {label_path}: {line}")
        cls_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
        labels.append([cls_id, *coords])
    return labels


def write_label_file(label_path: Path, labels):
    text = "\n".join(
        f"{int(cls_id)} {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}"
        for cls_id, cx, cy, box_w, box_h in labels
    )
    label_path.write_text(text + ("\n" if text else ""))


def find_image_for_label(images_dir: Path, label_path: Path):
    stem = label_path.stem
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def backup_dataset_files(data_root: Path, backup_root: Path):
    if backup_root.exists():
        return

    for split in ("train", "valid", "test"):
        labels_dir = data_root / split / "labels"
        target_dir = backup_root / split / "labels"
        target_dir.mkdir(parents=True, exist_ok=True)
        for label_path in labels_dir.glob("*.txt"):
            copy2(label_path, target_dir / label_path.name)

    data_yaml = data_root / "data.yaml"
    if data_yaml.exists():
        backup_root.mkdir(parents=True, exist_ok=True)
        copy2(data_yaml, backup_root / "data.yaml")


def classify_old_cup(label, image_shape, coco_boxes, coco_classes, min_iou: float):
    height, width = image_shape
    label_box = yolo_to_xyxy(label, width, height)
    ious = box_iou_one_to_many(label_box, coco_boxes)
    if len(ious) == 0:
        return 0, 0.0, None

    best_index = int(torch.argmax(ious).item())
    best_iou = float(ious[best_index].item())
    best_class = int(coco_classes[best_index].item())
    if best_iou < min_iou:
        return 0, best_iou, best_class

    if best_class == COCO_BOTTLE:
        return 1, best_iou, best_class
    return 0, best_iou, best_class


def update_data_yaml(data_root: Path, dry_run: bool):
    text = f"""path: {data_root.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 3
names:
  0: cup
  1: bottle
  2: mouse
"""
    if not dry_run:
        (data_root / "data.yaml").write_text(text)


def remove_label_caches(data_root: Path, dry_run: bool):
    for cache_path in data_root.glob("*/labels.cache"):
        if not dry_run:
            cache_path.unlink()


def main():
    args = parse_args()
    data_root = Path(args.data)
    backup_root = Path(args.backup)
    model = YOLO(args.model)

    if not args.dry_run:
        backup_dataset_files(data_root, backup_root)

    totals = Counter()
    changed_examples = []

    for split in ("train", "valid", "test"):
        labels_dir = data_root / split / "labels"
        images_dir = data_root / split / "images"
        for label_path in sorted(labels_dir.glob("*.txt")):
            labels = read_label_file(label_path)
            image_path = find_image_for_label(images_dir, label_path)

            coco_boxes = torch.empty((0, 4), dtype=torch.float32)
            coco_classes = torch.empty(0, dtype=torch.int64)
            image_shape = None
            if image_path and any(int(label[0]) == 0 for label in labels):
                result = model.predict(
                    str(image_path),
                    classes=[COCO_BOTTLE, COCO_CUP],
                    conf=args.coco_conf,
                    verbose=False,
                )[0]
                image_shape = result.orig_shape
                if result.boxes is not None and len(result.boxes) > 0:
                    coco_boxes = result.boxes.xyxy.cpu()
                    coco_classes = result.boxes.cls.cpu().int()

            new_labels = []
            file_changed = False
            for label in labels:
                old_class = int(label[0])
                if old_class == 1:
                    label[0] = 2
                    file_changed = True
                    totals["mouse_to_2"] += 1
                elif old_class == 0 and image_shape is not None:
                    new_class, best_iou, best_coco_class = classify_old_cup(
                        label, image_shape, coco_boxes, coco_classes, args.iou
                    )
                    if new_class == 1:
                        label[0] = 1
                        file_changed = True
                        totals["cup_to_bottle"] += 1
                        if len(changed_examples) < 30:
                            changed_examples.append(f"{split}: {image_path.name} IoU={best_iou:.2f}")
                    else:
                        totals["cup_kept"] += 1
                else:
                    totals[f"class_{old_class}_kept"] += 1
                new_labels.append(label)

            if file_changed and not args.dry_run:
                write_label_file(label_path, new_labels)

    update_data_yaml(data_root, args.dry_run)
    remove_label_caches(data_root, args.dry_run)

    print("Relabel summary")
    for key, value in totals.items():
        print(f"{key}: {value}")
    print("Changed bottle examples")
    for item in changed_examples:
        print(item)


if __name__ == "__main__":
    main()
