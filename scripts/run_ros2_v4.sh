#!/bin/sh
set -eu

if [ -f /opt/ros/humble/setup.sh ]; then
    . /opt/ros/humble/setup.sh
elif [ -f /opt/ros/foxy/setup.sh ]; then
    . /opt/ros/foxy/setup.sh
else
    echo "Could not find ROS2 setup file under /opt/ros."
    echo "Please install ROS2 first, or source your ROS2 setup file manually."
    exit 1
fi

cd "$(dirname "$0")/.."
/usr/bin/python3 src/ros2_realtime_detect.py --model models/cup_mouse_v4_yolov8s.pt --imgsz 320 --width 640 --height 480 --conf 0.4 "$@"
