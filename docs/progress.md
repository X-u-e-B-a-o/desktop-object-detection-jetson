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

## 2026-08-26
- 为提升模型对真实水壶、保温杯、高水杯的识别效果，将该类物体统一归入 `cup` 类。
- 使用 Roboflow 手动标注新的水壶/保温杯图片，并导出 YOLOv8 格式数据集 `Cup-Detection`。
- 新增水壶/保温杯数据：
  - 训练图片：22 张
  - 验证图片：9 张
  - cup 标注框：54 个
- 合并方式：
  - `Cup-Detection/train/images` 合并至 `data/combined/train/images`
  - `Cup-Detection/train/labels` 合并至 `data/combined/train/labels`
  - `Cup-Detection/valid/images` 合并至 `data/combined/valid/images`
  - `Cup-Detection/valid/labels` 合并至 `data/combined/valid/labels`
- 合并后数据集规模：
  - 训练图片：881 张
  - 训练标签文件：881 个
  - 验证图片：173 张
  - 验证标签文件：173 个
  - 测试图片：48 张
- 合并后 train + valid 标注框统计：
  - cup 标注框：1238 个
  - mouse 标注框：297 个
- 取消正在进行的 v3 训练，继续补充新的 cup 数据集后再统一训练。
- 新增 Roboflow 导出的 `cup` 数据集：
  - 原导出类别名为 `pink`
  - 因该数据集为单类杯子/水杯数据，合并时统一作为 `cup` 类处理
  - 训练图片：78 张
  - 验证图片：10 张
  - 测试图片：10 张
  - 标注框：98 个，类别编号均为 `0`
- 合并后完整数据集规模：
  - 训练图片：959 张
  - 训练标签文件：959 个
  - 验证图片：183 张
  - 验证标签文件：183 个
  - 测试图片：58 张
  - 测试标签文件：58 个
- 合并后全数据集标注框统计：
  - cup 标注框：1365 个
  - mouse 标注框：321 个
- 基于 `train_cup_mouse_v2_finetune/weights/best.pt` 继续微调训练第三版模型：

```bash
yolo detect train data=data/combined/data.yaml model=runs/detect/runs/train_cup_mouse_v2_finetune/weights/best.pt epochs=30 imgsz=640 batch=8 project=runs name=train_cup_mouse_v3_cup_more
```

- 第三版模型训练耗时：约 1.637 小时。
- 第三版模型权重保存位置：
  - `runs/detect/runs/train_cup_mouse_v3_cup_more/weights/best.pt`
  - `runs/detect/runs/train_cup_mouse_v3_cup_more/weights/last.pt`
- 第三版模型验证集结果：
  - Precision：0.935
  - Recall：0.847
  - mAP50：0.921
  - mAP50-95：0.654
- 第三版模型分类别验证结果：
  - cup：mAP50 = 0.935，mAP50-95 = 0.660
  - mouse：mAP50 = 0.906，mAP50-95 = 0.648
- 使用 test 集对第三版 `best.pt` 进行独立评估：

```bash
yolo detect val model=runs/detect/runs/train_cup_mouse_v3_cup_more/weights/best.pt data=data/combined/data.yaml split=test
```

- 第三版模型测试集结果：
  - Precision：0.913
  - Recall：0.944
  - mAP50：0.952
  - mAP50-95：0.739
- 第三版模型分类别测试结果：
  - cup：mAP50 = 0.932，mAP50-95 = 0.703
  - mouse：mAP50 = 0.971，mAP50-95 = 0.775
- 使用真实图片 `desk6.jpg` 测试第三版模型，在 `conf=0.3` 条件下检测到：
  - cup：0.682
  - mouse：0.895
  - mouse：0.715

## 2026-08-27
- 检查并清理合并后的 cup/mouse 数据集，确认数据集仍为 YOLO 检测格式：
  - 训练图片：959 张
  - 训练标签文件：959 个
  - 验证图片：183 张
  - 验证标签文件：183 个
  - 测试图片：58 张
  - 测试标签文件：58 个
- 检查标签文件：
  - 未发现非法类别编号
  - 未发现越界坐标
  - 已将少量 Roboflow 导出的分割格式标签转换为 YOLO 检测框格式
  - 已删除旧的 `labels.cache`，保证下一次训练重新扫描标签
- 为了减少真实场景中的误检和漏检，第四版模型从 YOLOv8s 预训练权重开始训练。
- 第四版模型训练命令：

```bash
yolo detect train data=data/combined/data.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=8 project=runs name=train_cup_mouse_v4_yolov8s
```

- 第四版模型训练耗时：约 7.595 小时。
- 第四版模型参数量：11,126,358。
- 第四版模型权重保存位置：
  - `runs/detect/runs/train_cup_mouse_v4_yolov8s/weights/best.pt`
  - `runs/detect/runs/train_cup_mouse_v4_yolov8s/weights/last.pt`
- 第四版模型验证集结果：
  - Precision：0.888
  - Recall：0.836
  - mAP50：0.895
  - mAP50-95：0.647
- 第四版模型分类别验证结果：
  - cup：Precision = 0.919，Recall = 0.878，mAP50 = 0.916，mAP50-95 = 0.661
  - mouse：Precision = 0.856，Recall = 0.795，mAP50 = 0.873，mAP50-95 = 0.633

- 训练完成后使用 test 集独立评估：

```bash
yolo detect val model=runs/detect/runs/train_cup_mouse_v4_yolov8s/weights/best.pt data=data/combined/data.yaml split=test
```

## 下一步计划
- 使用 test 集独立评估第四版模型
- 使用真实桌面图片测试第四版模型检测效果
- 编写摄像头实时检测脚本
- 将模型部署到 Jetson 开发板
- 后续根据实验需要考虑加入 phone 第三类
