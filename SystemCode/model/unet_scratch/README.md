# 独立 U-Net 从零训练

本目录自行实现四级 U-Net，不导入 E0–E4 模型代码，不加载已有或预训练权重。复用的是已核验的数据划分，不是历史实验权重。

本次运行：`runs/unet_scratch_20261016`，日志：`runs/unet_scratch_20261016.log`。

- 随机初始化，seed 42；32 起始通道，7,760,097 参数。
- RGB /255，输入 256×256；mask 最近邻缩放；同步水平/垂直随机翻转。
- 1160 张训练，290 张验证；读取前核对全部图片/标注 SHA256，检查训练与验证的 RGB 内容重叠。
- AdamW，学习率 0.0001，weight decay 0.0001，batch 8，BCE + soft Dice；固定 100 轮。
- 每轮验证，按验证集平均二值 Dice 保存 `best.pt`；`last.pt` 保存优化器和随机状态，支持恢复。
- 验证 Dice/IoU 在 256×256、阈值 0.5 下计算，不能直接与历史原尺寸测试指标比较。798 个测试条目未参与训练或模型选择。
- 边缘可由预测二值区域的轮廓获得；当前训练目标是区域分割，没有额外边缘监督。权重格式独立，不能直接交给 `unet_edges/predict.py`。

从 `SystemCode/model` 执行：

```bash
python3 unet_scratch/train.py --output runs/unet_scratch_20261016 --device mps
```

新训练要求输出目录不存在。意外中断后，确认旧进程已停止，再用相同命令加 `--resume` 从上一个完整 epoch 恢复；同一目录禁止并行训练。恢复检查源码、数据划分及配置是否一致，GPU 运算不保证跨平台位级一致。

查看进度：

```bash
tail -f runs/unet_scratch_20261016.log
cat runs/unet_scratch_20261016/status.json
```

`history.json` 保存每轮训练损失、验证损失、Dice、IoU 和耗时，`config.json` 保存设置与源码/划分哈希。后台运行由 caffeinate 防止空闲睡眠，但关机、退出登录或合盖仍可能中断训练。

依赖：Python 3.9+、PyTorch、NumPy、Pillow。本次实际使用 Python 3.9 / PyTorch 2.8.0 / NumPy 2.0.2 / Pillow 11.3.0。
