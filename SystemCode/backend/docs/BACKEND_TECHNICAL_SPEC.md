# EndoLens 录制视频分析后端技术交接文档

**版本：** 1.0（与冻结模型包 `unet-b-expanded-seed42-r98-v1` 对齐）  
**用途：** 供后端 Agent 按本契约实现或重构服务。本文定义的是当前可运行参考后端的行为边界；`docs/openapi.json` 是接口机器可读定义。

> 这是研究/教学联调服务，只分析录制视频。没有临床诊断、实时内镜、用户鉴权、病例隔离或公网部署能力。不得把 `foreground_score` 称为诊断置信度，也不得把 candidate frames 称为独立息肉数。

## 1. 目标和非目标

### 必须实现

- 上传受限的 H264 MP4，保存原视频并提供 Range 播放。
- 基于视频 PTS（而不是 `frame_index / fps`）抽样，异步逐帧推理。
- 保存每一个抽样帧的原始 RGB PNG、AI mask PNG、推理元数据和任务进度。
- 支持任务查询、取消、帧分页、关键帧、报告、人工 mask 复核及 CSV/JSON 导出。
- 使用给定的冻结模型，完整记录模型和后处理配置快照。

### 不应擅自加入或声称已具备

- 病灶跨帧跟踪、病灶去重、病例级诊断、毫米测量、实时 FPS 承诺。
- 换模型、TTA、AMP、min-max 标准化、letterbox、阈值搜索，或删除小区域。
- 将未抽样、失败帧或 `is_candidate=false` 标为“正常”。
- 在无明确设计的前提下让人工 mask 覆盖原 AI mask 或 AI 报告统计。

## 2. 当前交付物和推荐重构目录

当前包中已经可运行的实现：

```text
endolens_b_v1/
├── backend/
│   ├── app.py                 # 路由、FastAPI lifespan、单 worker Runtime
│   ├── video.py               # H264 MP4 校验、PTS 解码、抽样
│   └── storage.py             # SQLite records + 不可变二进制资产
├── endolens_model/
│   ├── adapter.py             # ModelAdapter：校验、预处理、推理、resize
│   ├── network.py             # ResNet34 U-Net 定义
│   └── postprocess.py         # 阈值、连通域、轮廓、候选帧规则
├── weights/unet_b_expanded_seed42.pt
├── model.json                 # 权重完整性与模型契约
├── docs/openapi.json          # 机器可读 API 契约
└── tests/                     # 模型和 HTTP/边界测试
```

若需要在正式项目内拆分，建议保持以下职责，而不是照搬单个 `app.py`：

```text
backend/
├── main.py                    # create_app、lifespan、CORS、路由注册
├── api/v1/
│   ├── health.py  models.py  videos.py  analyses.py  frames.py  exports.py
├── schemas/                   # Pydantic 请求/响应对象
├── services/
│   ├── analysis_service.py    # 任务状态机及报告
│   ├── inference_service.py   # ModelAdapter 生命周期及串行推理
│   ├── video_service.py       # PTS 抽样和解码
│   └── review_service.py      # 乐观并发控制的人工复核
├── repositories/              # task/video/frame/review/export 持久化
├── storage/                   # 文件资产与数据库实现
├── workers/                   # 队列消费循环；生产版可替换为持久队列
└── core/config.py             # 环境变量、限制、日志
```

## 3. 总体流程与职责边界

```text
客户端
  │  POST /videos (multipart MP4)
  ▼
VideoService ──校验 codec/PTS/分辨率──► Asset Storage + VideoRecord
  │
  │  POST /analyses
  ▼
AnalysisService ──创建 queued TaskRecord──► 单消费者队列
                                                │
                                                ▼
                          VideoService.schedule(PTS) → 顺序解码 RGB frame
                                                │
                                                ▼
                     ModelAdapter.predict → probability_map float32[H,W]
                                                │
                                                ▼
                       postprocess → AI mask / FrameRecord / PNG assets
                                                │
                                                ▼
                         前端轮询 status + frames + report / export
```

- **模型层**只接受单张 RGB `uint8[H,W,3]`，输出原尺寸 `float32[H,W]` 分数图。
- **视频层**负责时间戳、抽样、解码；模型层不认识视频、队列或 HTTP。
- **任务层**负责状态、取消、进度、资产落盘和错误隔离。
- **前端**读取已保存的帧记录；连续播放时没有精准同帧时必须隐藏 mask，不能把上一帧 mask 叠到新画面。

## 4. 模型接入契约（不可更改）

```python
from endolens_model import ModelAdapter, MODEL_VERSION

# 应仅在进程启动时创建一次；不要每帧重新加载 .pt。
adapter = ModelAdapter(model_dir="/absolute/path/endolens_b_v1", device="auto")
result = adapter.predict(frame_rgb_uint8, MODEL_VERSION)

probability = result.probability_map  # np.float32, [H, W], 值域 [0, 1]
```

输入必须是 RGB，而 OpenCV 默认输出 BGR，必须转换：

```python
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
```

模型固定约束：

| 项目 | 值 |
|---|---|
| 模型版本 | `unet-b-expanded-seed42-r98-v1` |
| 架构 | ImageNet 预训练来源的 ResNet34 encoder + 5-stage U-Net decoder |
| 输入 | RGB uint8 HWC，最大 8192 × 8192（视频服务另外限制 1080p） |
| 预处理 | Pillow bilinear 拉伸至 352×352，`/255`，ImageNet mean/std |
| 输出 | logits 双线性恢复原尺寸，再 sigmoid；`float32[H,W]` |
| 禁止 | letterbox、每图 minmax、AMP、TTA、融合、自动替换模型 |
| 完整性 | 启动时校验 `model.json` 指定的权重 SHA256 和来源 checkpoint SHA256 |

`inference_time_ms` 仅包括预处理、传输、前向、恢复原尺寸、sigmoid 和 CPU copy；不包括排队、视频解码或写文件。

## 5. 视频与任务处理规则

### 上传约束

- 仅 `.mp4` 文件、实际 MOV/MP4 容器、H264 视频流、正向、方形像素。
- 最大 100 MiB、时长 `0 < duration <= 900s`、最多 1080p、最多解码 60,000 帧。
- 必须存在严格递增的 PTS；旋转元数据、缺失 PTS、时间倒退均拒绝。
- 单次任务最多抽样 600 帧；超过时提示增加 interval 或缩短视频。

### 抽样与状态机

1. `POST /analyses` 写入 `queued` 任务并立即返回 `202`。
2. worker 标记 `running/indexing`，完整顺序解码一次，按 PTS 生成抽样目标。
3. 切为 `running/inference`，再次顺序解码；每帧以精确的 `frame_index + timestamp_seconds` 匹配目标。
4. 每个目标帧的推理错误只标记该帧 `failed`；任务继续执行。
5. 若全部失败则任务 `failed`；否则 `completed`。取消后为 `cancelled`。

状态：`queued → running → completed | failed | cancelled`。`target_frames` 和进度百分比在 indexing 阶段为 `null`，不是 0%。

当前参考实现只能 **一个进程、一个 Uvicorn worker、一个推理消费者**；`storage/server.lock` 会阻止多实例同时写同一个存储目录。生产扩展必须先引入持久队列、任务租约/幂等性和对象存储，不能简单增加 `--workers`。

## 6. 后处理与候选语义

对每张 `probability_map`：

1. `binary_mask = probability >= 0.5`；阈值固定为 0.5。
2. 8 邻域连通域，计算每个区域的像素面积、bbox、面积比例和区域平均前景分数。
3. PNG mask 始终保留全部前景，不按面积过滤。
4. `is_candidate = 存在区域且最大区域面积比例 >= min_region_area_ratio`。

默认 `ENDO_MIN_REGION_AREA_RATIO=0`。该面积门槛是未验证的工程演示规则，只影响 `is_candidate`，绝不可标为临床阈值。`foreground_score` 为最大连通域的平均前景分数；空 mask 为 `null`。

轮廓仅供显示：最多 100 个，单条最多 256 点，可能被简化或截断；mask PNG 才是像素级权威结果。

## 7. 持久化模型和资产布局

默认 `ENDO_STORAGE_DIR=runtime_data`，可替换成专用持久目录，不能指向训练 runs 目录。

```text
runtime_data/
├── records.sqlite3            # 表 records(kind, id, task, idx, data JSON)
├── server.lock                # 单进程所有权锁
├── videos/{video_id}.mp4
├── frames/{task_id}-f00000042.png
├── thumbs/{task_id}-f00000042.jpg
├── masks/{task_id}-f00000042.png
├── reviews/{revision_id}.png
└── exports/{export_id}.csv|json
```

记录类型与关键字段：

| kind | 主键 | 关键字段 |
|---|---|---|
| `video` | `video_id` | filename、video_url、stored_path、宽高、duration、fps、timestamp_origin_seconds |
| `analysis` | `task_id` | video_id、status、phase、configuration、计数、warnings、elapsed_seconds、cancel_requested |
| `frame` | `{task}-f{index:08d}` | 时间戳、尺寸、状态、AI URLs、候选/区域/分数、review 状态 |
| `review` | `revision_id` | frame_id、base_revision_id、review_status、edited_mask_url、notes |
| `export` | `export_id` | path、format |

`analysis.configuration` 是审计快照，必须保存：模型版本、权重和源 checkpoint hash、设备、预处理、分割阈值、面积门槛及后处理版本。任何模型升级都要新版本号和新权重，不得静默替换。

当前 `Store` 是 SQLite + 本地不可覆盖文件（`xb`）+ 约 2GiB 配额。服务重启时，`queued/running` 任务必须标为失败并保留已完成帧；不要假装恢复推理。

## 8. HTTP API 契约

Base URL：`/api/v1`。ID 均为 32 位小写十六进制。成功响应具体字段以 `docs/openapi.json` 与以下定义为准。

| 方法 | 路径 | 行为 |
|---|---|---|
| GET | `/health` | 服务、模型、设备状态；`clinical_use=false` |
| GET | `/models` | 唯一可用模型及后处理配置 |
| POST | `/videos` | 上传 `multipart/form-data` 的 `file`，返回 `201` VideoRecord |
| GET | `/videos/{video_id}/stream` | 返回 MP4，支持单 Range 请求 |
| POST | `/analyses` | 创建异步任务，返回 `202` |
| GET | `/analyses/{task_id}` | 任务状态与 progress/performance |
| POST | `/analyses/{task_id}/cancel` | 请求取消，返回 `202` |
| GET | `/analyses/{task_id}/frames` | 帧分页，可只取候选帧 |
| GET | `/analyses/{task_id}/keyframes` | 最多 60 个已完成帧的均匀抽样缩略项 |
| GET | `/assets/{kind}/{name}` | 读取 frames/thumbs/masks/reviews 静态资产 |
| PUT | `/frames/{frame_id}/review` | 上传人工 binary PNG 修订 |
| GET | `/analyses/{task_id}/report` | 任务完整汇总报告 |
| POST | `/analyses/{task_id}/exports` | 创建 CSV / JSON 导出 |
| GET | `/exports/{export_id}/download` | 下载导出文件 |

### 创建分析任务

```http
POST /api/v1/analyses
Content-Type: application/json

{
  "video_id": "0123456789abcdef0123456789abcdef",
  "sampling_mode": "interval",
  "interval_seconds": 1.0,
  "model_version": "unet-b-expanded-seed42-r98-v1"
}
```

- `sampling_mode`: `interval` 或 `every_frame`。
- `interval` 时 `interval_seconds` 必填，范围 `[0.1, 900]`；`every_frame` 时必须忽略并保存为 `null`。
- 只接受上述一个 `model_version`，其他版本返回 422。
- 队列已满返回 429。

### 查询任务（重要字段）

```json
{
  "task_id": "…",
  "status": "running",
  "phase": "inference",
  "target_frames": 120,
  "configuration": { "segmentation_threshold": 0.5 },
  "progress": {
    "target_frames": 120,
    "processed_frames": 30,
    "candidate_frames": 4,
    "failed_frames": 1,
    "percent": 25.8
  },
  "performance": { "elapsed_seconds": 12.3, "processing_fps": 2.44 },
  "warnings": ["RESEARCH_DEMO_NOT_CLINICAL"]
}
```

`processing_fps` 是成功抽样帧 / 总任务耗时，包含索引、解码、存储等开销，不是模型 FPS 或播放 FPS。

### 帧分页

`GET /analyses/{task_id}/frames?after=42&limit=100&candidate_only=false`

返回 `items`、`next_cursor`、`last_cursor`、`has_more`。客户端以 `frame_id` 去重、按 `frame_index` 排序，使用 `last_cursor` 轮询增量；切换 `candidate_only` 时必须重置 cursor。`frame_index` 是 0 基，时间显示必须以 `timestamp_seconds` 为准。

关键 `FrameRecord` 字段：

```json
{
  "frame_id": "{task_id}-f00000042",
  "frame_index": 42,
  "timestamp_seconds": 12.438,
  "width": 1920,
  "height": 1080,
  "status": "completed",
  "is_candidate": true,
  "source_frame_url": "/api/v1/assets/frames/…png",
  "mask_url": "/api/v1/assets/masks/…png",
  "foreground_score": 0.73,
  "segmented_area_ratio": 0.08,
  "contours": [],
  "regions": [],
  "review_status": "unreviewed",
  "revision_id": null
}
```

失败帧应有 `status="failed"`、`is_candidate=null` 和错误说明。它不等同于阴性结果。

### 人工复核

`PUT /frames/{frame_id}/review` 使用 multipart：

- `edited_mask`：必须与原图同尺寸、单通道、仅含 `0/1` 或 `0/255` 的 PNG，最大 16 MiB。
- `review_status`：`reviewed` 或 `rejected`。
- `notes`：最多 2000 字符。
- `base_revision_id`：首次为空，后续必须传当前 `revision_id`。

服务采用乐观锁：revision 不一致返回 `409`，客户端必须重新拉取帧并由用户决定是否再次保存。原始 `mask_url` 永不修改；人工结果放在 `edited_mask_url`。报告的 AI 统计只基于原 AI mask。

### 导出

```json
POST /api/v1/analyses/{task_id}/exports
{"format":"csv", "include_reviewed_masks":true}
```

只支持 `csv`、`json`。`include_reviewed_masks=true` 仅在导出中包括修订记录/URL，**不**打包 PNG 二进制文件。

## 9. 错误、限制与环境变量

| HTTP / 情况 | 处理 |
|---|---|
| 404 | ID、记录或资产不存在 |
| 409 | 不可复核失败帧，或 revision 冲突 |
| 413 | 上传超过 100MiB、复核 PNG 超过 16MiB、notes 太长 |
| 415 | 非 MP4 扩展名 |
| 422 | 视频编码/PTS/尺寸不合规，或 JSON/参数/模型版本无效 |
| 429 | 内存任务队列已满 |
| 单帧推理异常 | 保存为失败帧，继续任务 |
| 任务关键流程异常 | task 标记 `failed`，记录 `ExceptionType:message` |

| 环境变量 | 默认值 | 含义 |
|---|---|---|
| `ENDO_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` |
| `ENDO_STORAGE_DIR` | `runtime_data` | 本地持久化目录 |
| `ENDO_MIN_REGION_AREA_RATIO` | `0` | 未验证候选面积门槛，范围 0–1 |
| `ENDO_CORS_ORIGINS` | localhost:5173, 127.0.0.1:5173, localhost:3000 | 逗号分隔精确 Origin |

## 10. 运行、测试和交付验收

安装 Python 3.11、与系统匹配的 PyTorch 2.9.1 / torchvision 0.24.1 后：

```bash
python -m pip install -r requirements.txt
python verify_bundle.py
python run_tests.py
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
python examples/api_smoke.py
```

验收至少覆盖：

1. 上传合成 MP4 后能 Range 播放；非法编解码、旋转和超限视频被拒绝。
2. 创建任务立即返回，`indexing` 时进度为 null，完成后帧数与抽样目标一致。
3. 已完成帧具有同原图尺寸的 source PNG 和 mask PNG，且模型版本、配置快照完整。
4. 取消、单帧失败、服务重启、队列满、分页增量、CSV/JSON 导出均有确定结果。
5. 人工复核不会修改 AI mask；旧 revision 必须得到 409。
6. `GET /health` 明确返回当前实际设备，且警示信息仍在所有分析结果中。

## 11. 上线前必须另行设计的事项

该原型不应直接暴露公网。若要扩展，应分别做设计和安全评审：认证授权与多租户隔离、TLS/反向代理上传限制、恶意媒体解码隔离、对象存储与数据生命周期、数据库迁移、可观测性、持久任务队列、跨进程模型资源管理、速率限制、审计和备份。

模型侧还需要独立验证无息肉视频的安全性、候选面积阈值、跨帧病灶定义和任何临床用途；工程实现本身不能替代这些验证。
