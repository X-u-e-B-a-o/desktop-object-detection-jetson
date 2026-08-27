from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models/cup_mouse_v4_yolov8s.pt",
    PROJECT_ROOT / "runs/detect/runs/train_cup_mouse_v4_yolov8s/weights/best.pt",
    PROJECT_ROOT / "runs/detect/runs/train_cup_mouse_v3_cup_more/weights/best.pt",
]


def default_model_path() -> str:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return str(path)
    return str(MODEL_CANDIDATES[0])


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def parse_args():
    parser = ArgumentParser(description="Run real-time YOLO cup/mouse detection from a camera or video.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or OpenCV/GStreamer camera source.")
    parser.add_argument("--model", default=default_model_path(), help="Path to YOLO model weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.3, help="Base confidence threshold.")
    parser.add_argument("--cup-conf", type=float, default=0.3, help="Minimum confidence to keep cup detections.")
    parser.add_argument("--mouse-conf", type=float, default=0.3, help="Minimum confidence to keep mouse detections.")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height.")
    parser.add_argument("--no-window", action="store_true", help="Run without opening a display window.")
    parser.add_argument("--print", action="store_true", help="Print detections to terminal.")
    return parser.parse_args()


def keep_class_thresholds(result, cup_conf: float, mouse_conf: float):
    if result.boxes is None or len(result.boxes) == 0:
        return result

    data = result.boxes.data
    conf = data[:, -2]
    cls = data[:, -1].int()
    keep = ((cls == 0) & (conf >= cup_conf)) | ((cls == 1) & (conf >= mouse_conf))
    result.boxes = Boxes(data[keep], result.orig_shape)
    return result


def open_capture(source, width: int, height: int):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def print_detections(result):
    if result.boxes is None or len(result.boxes) == 0:
        print("no detections")
        return

    names = result.names
    parts = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        parts.append(f"{names[cls_id]} {conf:.2f}")
    print(", ".join(parts))


def main():
    args = parse_args()
    model = YOLO(args.model)
    source = parse_source(args.source)
    cap = open_capture(source, args.width, args.height)

    window_name = "cup-mouse realtime detection"
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
                device=args.device,
                verbose=False,
            )
            result = keep_class_thresholds(results[0], args.cup_conf, args.mouse_conf)

            now = perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0 else (0.9 * fps + 0.1 * current_fps)

            if args.print:
                print_detections(result)

            if not args.no_window:
                annotated = result.plot()
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
