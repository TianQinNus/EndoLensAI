# 息肉轮廓图片工具

自包含模型包：输入一张照片，输出原尺寸 PNG，只有绿色预测 mask 边缘，不填充区域，不绘制矩形框。默认线宽 2 像素，可选 1–20。多区域和内部孔洞的边缘均保留；预测为空则输出原图。按 EXIF 方向显示照片，输出转为 RGB。

## 使用

需要 Python 3.9 或以上。首次安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

在本文件所在目录运行：

```bash
python3 predict.py /绝对路径/照片.jpg -o /绝对路径/标注照片.png
```

不指定 `-o` 时，在输入照片旁生成 `原文件名_outline.png`。输出已存在时拒绝覆盖。可添加 `--line-width 3` 调整轮廓宽度。

默认 CPU，避免与后台训练争用 GPU。训练停止后可显式使用 `--device mps`（Mac）或 `--device cuda`（CUDA 环境）。只需本目录中的文件，无需训练数据或原工程。

## Python 调用

从本目录调用：

```python
from predict import PolypOutline
model = PolypOutline()  # 只加载一次，可重复处理照片
model.predict('photo.jpg', 'photo_outline.png')
```

也可调用 `annotate(PIL_image)` 返回 PIL 图片，不保存文件。

## 模型与协议

采用已完成实验中验证 Dice 最高的 `unetpp_context_20261020` 第 100 轮权重：验证 Dice 0.893595。798 张已知评估图片平均 Dice 0.718374，IoU 0.638657；这些图片不是独立最终留出集。曾报告的最高测试 Dice 属于其他模型，本包仍遵循验证集选择规则。

RGB /255，输入双线性缩放至 320×320；输出 logits 双线性恢复原图尺寸（align_corners=False），sigmoid 后固定阈值 0.5；二值 mask 减去腐蚀结果提取内侧边缘。轮廓是模型预测，不是真实标注。

`best.pt` 是已训练权重，`model.py` 是独立推理结构，`model_info.json` 保存来源和 SHA256。加载时核对权重及模型源码哈希。全部训练从随机初始化开始，没有外部预训练权重。
