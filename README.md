# Desktop Object Detection on Jetson

基于 YOLO 和 ROS2 的 Jetson 桌面目标检测与识别实验。

## 项目目标

- 自行采集并标注桌面物体数据集
- 训练目标检测模型
- 在 Jetson 上实时运行识别程序
- 显示目标类别、检测框和置信度
- 通过 ROS2 发布识别结果

## 检测类别

当前模型类别：
- cup
- mouse

后续可扩展：
- phone

## 技术路线

数据采集 -> 数据标注 -> YOLO 训练 -> 摄像头实时检测 -> Jetson 部署 -> ROS2 发布结果

## 当前进展

- 已完成 cup 与 mouse 两类 YOLO 数据集合并。
- 已使用 YOLOv8n 训练第一版目标检测模型。
- 已使用 test 集完成独立评估。
- 已补充真实桌面场景图片，并用 Roboflow 手动标注后合并进训练集，用于后续微调模型。
- 已将水壶、保温杯等高水杯类物体统一归入 `cup` 类。
- 准备使用 YOLOv8s 预训练模型重新训练 50 个 epoch，提升真实场景下的稳定性。

## 数据集说明

数据集采用 YOLO 目标检测格式，每张图片对应一个 `.txt` 标签文件。标签文件中每一行表示一个目标，包括类别编号、目标中心点坐标以及边界框宽高。

当前类别编号：

```text
0 = cup
1 = mouse
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
water bottle / thermos as cup: 31 images
additional cup images: 98 images
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

## 项目进度

见 `docs/progress.md`
