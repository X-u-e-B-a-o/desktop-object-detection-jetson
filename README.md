# Desktop Object Detection on Jetson

基于 YOLO 和 ROS2 的 Jetson 桌面目标检测与识别实验。

## 项目目标

- 自行采集并标注桌面物体数据集
- 训练目标检测模型
- 在 Jetson 上实时运行识别程序
- 显示目标类别、检测框和置信度
- 通过 ROS2 发布识别结果

## 检测类别

当前 Jetson 演示模型类别：
- cup
- mouse

后续可扩展：
- bottle
- phone

## 技术路线

数据采集 -> 数据标注 -> YOLO 训练 -> 摄像头实时检测 -> Jetson 部署 -> ROS2 发布结果

## 当前进展

- 已完成 cup 与 mouse 两类 YOLO 数据集合并。
- 已使用 YOLOv8n 训练第一版目标检测模型。
- 已使用 test 集完成独立评估。
- 已补充真实桌面场景图片，并用 Roboflow 手动标注后合并进训练集，用于后续微调模型。
- 已完成 YOLOv8s 预训练模型 50 个 epoch 训练和 test 集评估。
- 已将第四版 YOLOv8s 模型部署到 Jetson，并补充 ROS2 实时检测节点。

## 数据集说明

数据集采用 YOLO 目标检测格式，每张图片对应一个 `.txt` 标签文件。标签文件中每一行表示一个目标，包括类别编号、目标中心点坐标以及边界框宽高。

当前整理后的数据集类别编号：

```text
0 = cup
1 = bottle
2 = mouse
```

当前合并后的训练集规模：

```text
train images: 959
valid images: 183
test images: 58
```

其中已补充真实桌面场景数据：

```text
cup/mouse real images: 34 images
water bottle / thermos images: 31 images
additional cup images: 98 images
```

当前三分类标注框统计：

```text
cup annotations: 1236
bottle annotations: 129
mouse annotations: 321
```

出于仓库体积和隐私考虑，原始数据集、真实测试图片、训练输出和模型权重不直接提交到 GitHub。

## 模型结果

已完成 YOLOv8n 第一版模型训练 30 个 epoch，验证集和测试集结果如下：

```text
Validation mAP50: 0.922
Test mAP50: 0.959
```

已使用 YOLOv8s 预训练权重重新训练第四版模型 50 个 epoch：

```bash
yolo detect train data=data/combined/data.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=8 project=runs name=train_cup_mouse_v4_yolov8s
```

第四版模型验证集结果：

```text
Validation mAP50: 0.895
Validation mAP50-95: 0.647
Test mAP50: 0.960
Test mAP50-95: 0.722
```

详细训练命令、评估结果和进度记录见 `docs/progress.md`。

## 真实图片测试

真实桌面图片可放在本地 `test_images/` 目录中。该目录不会提交到 GitHub。

推荐使用项目脚本进行预测，默认会过滤低置信度框并减少重复框：

```bash
python src/predict_image.py test_images/
```

测试单张图片：

```bash
python src/predict_image.py test_images/desk4.jpg
```

如果仍然出现较多误检，可以提高置信度阈值：

```bash
python src/predict_image.py test_images/desk4.jpg --conf 0.7
```

## Jetson 部署与实时检测

训练完成后的模型权重不提交到 GitHub，需要单独传到 Jetson。推荐把第四版模型复制为：

```text
models/cup_mouse_v4_yolov8s.pt
```

从 Mac 传到 Jetson 的示例命令：

```bash
scp runs/detect/runs/train_cup_mouse_v4_yolov8s/weights/best.pt jetson@JETSON_IP:~/desktop-object-detection-jetson/models/cup_mouse_v4_yolov8s.pt
```

在 Jetson 上进入项目目录后安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python numpy
```

接 USB 摄像头实时检测：

```bash
python src/realtime_detect.py --source 0 --model models/cup_mouse_v4_yolov8s.pt --conf 0.3 --cup-conf 0.3 --mouse-conf 0.3
```

运行后会打开摄像头窗口并实时显示检测框，按 `q` 退出。

实时检测窗口中可使用快捷键：

```text
s = 保存当前检测截图
q = 退出实时检测
```

截图默认保存到：

```text
~/xb/realtime_frames/
```

如果自训练模型在复杂背景中误检较多，可以使用 COCO 预训练模型做演示版实时检测。该脚本保留 YOLOv8s 原始 80 类判断能力，但只显示 `bottle`、`cup`、`mouse`：

```bash
python src/realtime_coco_filter.py --source 0 --model yolov8s.pt --conf 0.3
```

## ROS2 实时检测节点

为满足实验中 ROS2 通信要求，项目新增 ROS2 实时检测脚本：

```text
src/ros2_realtime_detect.py
```

该节点会在 Jetson 上完成：

```text
摄像头采集 -> YOLO 推理 -> OpenCV 实时显示 -> ROS2 发布图像和检测结果
```

为保证 Jetson 上实时帧率，默认只发布检测结果 topic：

```text
/yolo/detections       检测结果 JSON 字符串，类型 std_msgs/String
```

如需同时发布图像 topic，可在运行命令中加入 `--publish-images`：

```text
/camera/image_raw      原始摄像头图像，类型 sensor_msgs/Image
/yolo/annotated_image  带检测框的图像，类型 sensor_msgs/Image
```

Jetson 上运行前先加载 ROS2 环境：

```bash
source /opt/ros/humble/setup.bash
```

运行 v4 模型的 ROS2 实时检测：

```bash
python3 src/ros2_realtime_detect.py --model models/cup_mouse_v4_yolov8s.pt --conf 0.4
```

Jetson 帧率不足时推荐使用轻量参数：

```bash
python3 src/ros2_realtime_detect.py --model models/cup_mouse_v4_yolov8s.pt --imgsz 320 --width 640 --height 480 --conf 0.4
```

也可以使用项目提供的启动脚本：

```bash
./scripts/run_ros2_v4.sh
```

查看检测结果 topic：

```bash
ros2 topic echo /yolo/detections
```

查看当前发布的 topic：

```bash
ros2 topic list
```

## 项目进度

见 `docs/progress.md`
