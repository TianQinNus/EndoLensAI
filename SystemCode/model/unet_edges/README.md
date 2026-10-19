# U-Net 息肉区域与边缘识别

复用冻结的 `e1_mac/unet.py` 和 `e1_mac/run_e1.py`，新增单张图片的边缘推理入口。模型预测区域，边缘由二值区域减去 3×3 腐蚀得到；不是独立的边缘监督模型。红色线表示预测轮廓。

所有命令从 `SystemCode/model` 运行。正式环境依赖见 `e1_mac/requirements-e1-mac.txt`，推荐 Python 3.11。无需下载预训练权重。

## 数据读取与模型

- 已有 `dataset/TrainDataset/images` 与 `masks`，按同名文件配对；忽略 `.DS_Store`。
- RGB 双线性缩放，ImageNet 均值/标准差归一化；mask 最近邻缩放；图像和标注同步随机翻转。
- 四级 U-Net、跳跃连接、单通道 logits；BCE + soft Dice loss，AdamW。
- 正式 E1 使用全部 1450 对训练数据，固定训练 100 轮，使用最后一轮权重。测试集不参与选轮次或调阈值；本流程不提供内部验证早停。
- 推理先将 logits 恢复原图大小，再 sigmoid、固定 0.5 阈值，最后提取边缘。

## 核验及试跑

```bash
python3 scripts/bootstrap_artifacts.py
python3 e1_mac/run_e1.py audit --output runs/unet_data_audit
python3 e1_mac/run_e1.py train --smoke --input-size 128 --base-channels 16 --device cpu --output runs/unet_edge_smoke --log-every 1
python3 unet_edges/test_boundary.py
```

输出目录必须不存在，重复运行时使用新名字。Smoke 仅使用 8 张训练图片训练 1 轮、评估 5 张图片，用于验证端到端流程，不代表识别效果。

## Mac 正式训练

```bash
python3 e1_mac/check_device.py
python3 e1_mac/run_e1.py train --device mps --epochs 100 --input-size 352 --base-channels 64 --batch-size 2 --accumulation 4 --train-only --output runs/unet_full
python3 e1_mac/run_e1.py evaluate --run-dir runs/unet_full --device cpu
```

MPS 不可用时可以显式用 `--device cpu`，但完整模型训练耗时较长。恢复训练：使用相同训练参数，将 `--output runs/unet_full` 换成 `--resume runs/unet_full`。正式训练尚未因本次试跑而完成。

## 输出息肉边缘

```bash
python3 unet_edges/predict.py --checkpoint runs/unet_full/checkpoints/unet_final.pth --image dataset/TrainDataset/images/348.png --output runs/unet_edge_prediction --device auto
```

输出 `probability.npz`、`mask.png`、`edge.png`、`overlay.png` 和含数据/权重/脚本哈希的 `prediction.json`。入口仅支持本仓库 E1 Mac 格式及其源码哈希匹配的权重。可以替换为 smoke 权重检查流程，但不能据此判断分割质量。
