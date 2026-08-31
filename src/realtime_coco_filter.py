from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "yolov8s.pt"

COCO_BOTTLE = 39
COCO_CUP = 41
COCO_MOUSE = 64
TARGET_CLASSES = [COCO_BOTTLE, COCO_CUP, COCO_MOUSE]
DISPLAY_NAMES = {
    COCO_BOTTLE: "cup",
    COCO_CUP: "cup",
    COCO_MOUSE: "mouse",
}
COLORS = {
    "cup": (255, 0, 0),
    "mouse": (255, 255, 0),
}


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def parse_args():
    parser = ArgumentParser(description="Run real-time detection with pretrained COCO YOLO and display only cup/mouse.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or OpenCV/GStreamer camera source.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to pretrained YOLO model weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height.")
    parser.add_argument("--no-window", action="store_true", help="Run without opening a display window.")
    parser.add_argument("--print", action="store_true", help="Print detections to terminal.")
    return parser.parse_args()


def open_capture(source, width: int, height: int):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def draw_filtered_boxes(frame, result):
    detections = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return frame, detections

    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id not in DISPLAY_NAMES:
            continue

        display_name = DISPLAY_NAMES[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
        color = COLORS[display_name]
        label = f"{display_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 28)), (x1 + 150, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        detections.append(label)

    return frame, detections


def main():
    args = parse_args()
    model = YOLO(args.model)
    source = parse_source(args.source)
    cap = open_capture(source, args.width, args.height)

    window_name = "COCO filtered cup/mouse realtime detection"
    last_time = perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                classes=TARGET_CLASSES,
                device=args.device,
                verbose=False,
            )

            now = perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0 else (0.9 * fps + 0.1 * current_fps)

            annotated, detections = draw_filtered_boxes(frame, results[0])
            if args.print:
                print(", ".join(detections) if detections else "no detections")

            if not args.no_window:
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
