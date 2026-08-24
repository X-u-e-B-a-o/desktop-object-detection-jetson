# Desktop Object Detection on Jetson

基于 YOLO 和 ROS2 的 Jetson 桌面目标检测与识别实验。

## 项目目标

- 自行采集并标注桌面物体数据集
- 训练目标检测模型
- 在 Jetson 上实时运行识别程序
- 显示目标类别、检测框和置信度
- 通过 ROS2 发布识别结果

## 检测类别

暂定：
- cup
- mouse
- phone

## 技术路线

数据采集 -> 数据标注 -> YOLO 训练 -> 摄像头实时检测 -> Jetson 部署 -> ROS2 发布结果

## 项目进度

见 `docs/progress.md`