# 项目进度记录

## 2026-08-24
- 创建 GitHub 仓库
- 创建 PyCharm 本地项目
- 绑定本地项目与 GitHub 仓库
- 确定项目技术路线：Python + YOLO + OpenCV + ROS2 + Jetson
- 初步确定检测类别：cup、mouse、phone

## 2026-08-25
- 合并 cup 与 mouse 两类 YOLO 数据集，生成统一训练数据集：
  - 数据集路径：`data/combined/data.yaml`
  - 类别编号：`0 = cup`，`1 = mouse`
  - 训练集：825 张图像
  - 验证集：164 张图像
  - 测试集：48 张图像
- 使用 YOLOv8n 训练第一版两类目标检测模型：

```bash
yolo detect train data=data/combined/data.yaml model=yolov8n.pt epochs=30 imgsz=640 batch=8 project=runs name=train_cup_mouse_v1
```

- 训练环境：Mac，Apple M3，CPU 训练。
- 训练耗时：约 2.024 小时。
- 模型权重保存位置：
  - `runs/detect/runs/train_cup_mouse_v1/weights/best.pt`
  - `runs/detect/runs/train_cup_mouse_v1/weights/last.pt`
- 验证集最终结果：
  - Precision：0.931
  - Recall：0.850
  - mAP50：0.922
  - mAP50-95：0.640
- 分类别验证结果：
  - cup：mAP50 = 0.923，mAP50-95 = 0.646
  - mouse：mAP50 = 0.921，mAP50-95 = 0.635
- 使用 test 集对 `best.pt` 进行独立评估：

```bash
yolo detect val model=runs/detect/runs/train_cup_mouse_v1/weights/best.pt data=data/combined/data.yaml split=test
```

- 测试集结果：
  - Precision：0.888
  - Recall：0.948
  - mAP50：0.959
  - mAP50-95：0.709
- 分类别测试结果：
  - cup：mAP50 = 0.952，mAP50-95 = 0.651
  - mouse：mAP50 = 0.966，mAP50-95 = 0.766
- 当前结论：第一版 cup + mouse 模型已经训练完成，验证集和测试集效果均较好，可以进入真实图片测试和 Jetson 部署准备阶段。
- 使用真实桌面照片测试时发现部分复杂场景存在误检和漏检，例如：
  - 水瓶被误检为 cup
  - 玻璃杯被误检为 mouse
  - 黑色鼠标在复杂背景下置信度偏低或漏检
- 为改善真实场景效果，使用 Roboflow 手动标注新拍摄的 cup/mouse 图片，并将导出的 YOLOv8 数据集合并进训练集。
- 本次新增真实场景训练数据：
  - 新增训练图片：34 张
  - 新增标注框：34 个
  - 新增 cup 标注：21 个
  - 新增 mouse 标注：13 个
- 合并后训练集规模：
  - 训练图片：859 张
  - 训练标签文件：859 个
  - 训练集标注框总数：1257 个
  - cup 标注总数：1035 个
  - mouse 标注总数：222 个

## 下一步计划
- 基于 `train_cup_mouse_v1/weights/best.pt` 微调训练第二版模型
- 使用真实桌面图片测试第二版模型检测效果
- 编写摄像头实时检测脚本
- 将模型部署到 Jetson 开发板
- 后续根据实验需要考虑加入 phone 第三类
