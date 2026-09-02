from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models/cup_bottle_mouse_v5_yolov8s.pt",
    PROJECT_ROOT / "runs/detect/runs/train_cup_bottle_mouse_v5_yolov8s/weights/best.pt",
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
    parser = ArgumentParser(description="Run real-time YOLO cup/bottle/mouse detection from a camera or video.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or OpenCV/GStreamer camera source.")
    parser.add_argument("--model", default=default_model_path(), help="Path to YOLO model weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.3, help="Base confidence threshold.")
    parser.add_argument("--cup-conf", type=float, default=0.3, help="Minimum confidence to keep cup detections.")
    parser.add_argument("--bottle-conf", type=float, default=0.3, help="Minimum confidence to keep bottle detections.")
    parser.add_argument("--mouse-conf", type=float, default=0.3, help="Minimum confidence to keep mouse detections.")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height.")
    parser.add_argument("--display-scale", type=float, default=1.5, help="Scale the display window without changing inference.")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Limit processing to this target FPS.")
    parser.add_argument("--save-frames", action="store_true", help="Deprecated. Press s in the window to save one frame.")
    parser.add_argument(
        "--save-dir",
        default=str(Path.home() / "xb/realtime_frames"),
        help="Base folder for screenshots saved with the s key. A timestamped folder is created inside it.",
    )
    parser.add_argument("--max-empty-frames", type=int, default=30, help="Stop after this many failed frame reads.")
    parser.add_argument("--scan-cameras", action="store_true", help="Scan camera indexes and exit.")
    parser.add_argument("--no-window", action="store_true", help="Run without opening a display window.")
    parser.add_argument("--print", action="store_true", help="Print detections to terminal.")
    return parser.parse_args()


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


def open_capture(source, width: int, height: int):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def scan_cameras(max_index: int = 5):
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        ok = cap.isOpened()
        frame_ok = False
        shape = None
        if ok:
            frame_ok, frame = cap.read()
            if frame_ok:
                shape = frame.shape
        cap.release()
        status = "available" if ok and frame_ok else "unavailable"
        print(f"camera {index}: {status}" + (f", frame_shape={shape}" if shape else ""))


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


def create_frame_output_dir(save_dir: str) -> Path:
    output_dir = Path(save_dir).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving frames to: {output_dir}")
    return output_dir


def scale_for_display(frame, scale: float):
    if scale <= 0 or abs(scale - 1.0) < 0.01:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()
    if args.scan_cameras:
        scan_cameras()
        return

    model = YOLO(args.model)
    source = parse_source(args.source)
    cap = open_capture(source, args.width, args.height)

    window_name = "cup-bottle-mouse realtime detection"
    last_time = perf_counter()
    fps = 0.0
    empty_frames = 0
    frame_index = 0
    save_dir = None
    frame_interval = 1.0 / args.target_fps if args.target_fps and args.target_fps > 0 else 0.0
    if args.save_frames:
        print("Press s in the detection window to save one screenshot. Automatic frame saving is disabled.")

    try:
        while True:
            loop_start = perf_counter()
            ok, frame = cap.read()
            if not ok:
                empty_frames += 1
                if empty_frames >= args.max_empty_frames:
                    print("Could not read frame from camera.")
                    break
                sleep(0.03)
                continue
            empty_frames = 0

            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            result = keep_class_thresholds(results[0], args.cup_conf, args.bottle_conf, args.mouse_conf)

            now = perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0 else (0.9 * fps + 0.1 * current_fps)

            if args.print:
                print_detections(result)

            annotated = None
            if not args.no_window or save_dir is not None:
                annotated = result.plot()
                cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if not args.no_window and annotated is not None:
                display_frame = scale_for_display(annotated, args.display_scale)
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    if save_dir is None:
                        save_dir = create_frame_output_dir(args.save_dir)
                    frame_index += 1
                    screenshot_path = save_dir / f"screenshot_{frame_index:03d}.jpg"
                    cv2.imwrite(str(screenshot_path), display_frame)
                    print(f"Saved screenshot: {screenshot_path}")

            if frame_interval > 0:
                elapsed = perf_counter() - loop_start
                if elapsed < frame_interval:
                    sleep(frame_interval - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
