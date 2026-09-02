from argparse import ArgumentParser
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter, sleep

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from ultralytics.engine.results import Boxes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models/cup_mouse_v4_yolov8s.pt",
    PROJECT_ROOT / "models/cup_mouse_v3.pt",
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
    parser = ArgumentParser(description="Run YOLO realtime detection and publish ROS2 topics.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or OpenCV/GStreamer source.")
    parser.add_argument("--model", default=default_model_path(), help="Path to YOLO model weights.")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.4, help="Base confidence threshold.")
    parser.add_argument("--cup-conf", type=float, default=0.4, help="Minimum confidence for cup.")
    parser.add_argument("--bottle-conf", type=float, default=0.4, help="Minimum confidence for bottle.")
    parser.add_argument("--mouse-conf", type=float, default=0.4, help="Minimum confidence for mouse.")
    parser.add_argument("--iou", type=float, default=0.4, help="NMS IoU threshold.")
    parser.add_argument("--device", default=None, help="Inference device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height.")
    parser.add_argument("--display-scale", type=float, default=1.5, help="Scale the display window without changing inference.")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Limit processing to this target FPS.")
    parser.add_argument("--frame-id", default="camera", help="ROS frame_id used in image headers.")
    parser.add_argument("--publish-images", action="store_true", help="Also publish raw and annotated image topics.")
    parser.add_argument("--no-window", action="store_true", help="Run without opening an OpenCV display window.")
    parser.add_argument("--save-dir", default=str(Path.home() / "xb/ros2_screenshots"), help="Folder for screenshots.")
    return parser.parse_args()


def open_capture(source, width: int, height: int):
    if isinstance(source, int) and sys.platform == "darwin":
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


def keep_class_thresholds(result, cup_conf: float, bottle_conf: float, mouse_conf: float):
    if result.boxes is None or len(result.boxes) == 0:
        return result

    data = result.boxes.data
    conf = data[:, -2]
    cls = data[:, -1].int()
    thresholds = {"cup": cup_conf, "bottle": bottle_conf, "mouse": mouse_conf}
    keep = cls == -1
    for cls_id, name in result.names.items():
        if name in thresholds:
            keep |= (cls == int(cls_id)) & (conf >= thresholds[name])
    result.boxes = Boxes(data[keep], result.orig_shape)
    return result


def image_msg_from_frame(node: Node, frame, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = frame.tobytes()
    return msg


def detections_json(result) -> str:
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [round(float(value), 2) for value in box.xyxy[0]]
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            )
    return json.dumps({"detections": detections}, ensure_ascii=False)


def create_screenshot_dir(save_dir: str) -> Path:
    output_dir = Path(save_dir).expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving screenshots to: {output_dir}")
    return output_dir


def scale_for_display(frame, scale: float):
    if scale <= 0 or abs(scale - 1.0) < 0.01:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def main():
    args = parse_args()
    rclpy.init()
    node = Node("yolo_realtime_detector")
    raw_pub = node.create_publisher(Image, "/camera/image_raw", 1) if args.publish_images else None
    annotated_pub = node.create_publisher(Image, "/yolo/annotated_image", 1) if args.publish_images else None
    detections_pub = node.create_publisher(String, "/yolo/detections", 1)

    model = YOLO(args.model)
    cap = open_capture(parse_source(args.source), args.width, args.height)

    window_name = "ROS2 YOLO realtime detection"
    frame_interval = 1.0 / args.target_fps if args.target_fps and args.target_fps > 0 else 0.0
    last_time = perf_counter()
    fps = 0.0
    screenshot_dir = None
    screenshot_index = 0

    print("Publishing ROS2 topic: /yolo/detections")
    if args.publish_images:
        print("Publishing ROS2 image topics: /camera/image_raw, /yolo/annotated_image")
    print("Press s to save a screenshot, q to quit.")

    try:
        while rclpy.ok():
            loop_start = perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from camera.")
                break

            if raw_pub is not None:
                raw_pub.publish(image_msg_from_frame(node, frame, args.frame_id))
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
            result = keep_class_thresholds(result, args.cup_conf, args.bottle_conf, args.mouse_conf)

            annotated = result.plot() if not args.no_window or annotated_pub is not None else None
            now = perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = current_fps if fps == 0 else (0.9 * fps + 0.1 * current_fps)
            if annotated is not None:
                cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            detections_pub.publish(String(data=detections_json(result)))
            if annotated_pub is not None and annotated is not None:
                annotated_pub.publish(image_msg_from_frame(node, annotated, args.frame_id))
            rclpy.spin_once(node, timeout_sec=0)

            if not args.no_window and annotated is not None:
                display_frame = scale_for_display(annotated, args.display_scale)
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    if screenshot_dir is None:
                        screenshot_dir = create_screenshot_dir(args.save_dir)
                    screenshot_index += 1
                    path = screenshot_dir / f"screenshot_{screenshot_index:03d}.jpg"
                    cv2.imwrite(str(path), display_frame)
                    print(f"Saved screenshot: {path}")

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
