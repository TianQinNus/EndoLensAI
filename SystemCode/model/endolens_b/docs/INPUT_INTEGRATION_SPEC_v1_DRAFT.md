# EndoLens AI — Technical Integration Specification

**Version:** 1.0  
**Status:** Proposed — Pending Team Confirmation  
**Project:** EndoLens AI / PolypLens AI  
**Purpose:** Model, Backend, and Frontend Integration

---

## 1. Project Overview

EndoLens AI is a web-based system for AI-assisted colonoscopy video analysis.

The system allows users to:

1. Upload a recorded colonoscopy video.
2. Configure the frame sampling strategy.
3. Start an AI segmentation task.
4. Monitor analysis progress.
5. View candidate frames and segmentation masks.
6. Navigate the video using timestamps and keyframes.
7. Manually edit segmentation masks.
8. Generate and export analysis reports.

The first release supports **recorded-video analysis**, not live endoscope integration.

The system consists of three components:

| Component | Input | Output | Responsibility |
|---|---|---|---|
| Model | RGB image frame | Foreground probability map | Polyp segmentation |
| Backend | Video, configuration, model output, user actions | JSON responses, masks, thumbnails, reports | Processing, storage, APIs |
| Frontend | User input, API responses | Interactive visualization and review actions | User interface |

### 1.1 End-to-End Data Flow

```text
User
 |
 | Upload MP4 Video
 v
React Frontend
 |
 | POST /api/v1/videos
 v
FastAPI Backend
 |
 | Decode Video
 | Extract Frames
 | Preserve Frame Index & Timestamp
 v
RGB Frame (H, W, 3)
 |
 | Model Inference
 v
Probability Map (H, W)
 |
 | Threshold + Connected Components
 v
Binary Mask + Candidate Flag
 |
 | Generate Contours + Thumbnails
 | Save Frame Results
 v
Backend Storage
 |
 | GET Analysis / Frames APIs
 v
React Frontend
 |
 +-- Video Player
 +-- Segmentation Overlay
 +-- Detection Panel
 +-- Video Timeline
 +-- Keyframes
 +-- Mask Editor
 +-- Reports
```

---

# 2. Model → Backend Integration

## 2.1 Model Input

The backend extracts frames from uploaded videos and sends them to the segmentation model.

### Input Specification

| Field | Type | Required | Description |
|---|---|---|---|
| `frame` | `np.ndarray` | Yes | RGB image |
| `height` | integer | Derived | Original frame height |
| `width` | integer | Derived | Original frame width |
| `model_version` | string | Yes | Selected model configuration |

**Frame format:**

```python
frame.shape == (H, W, 3)
frame.dtype == np.uint8
```

Color format: RGB.

Example:

```python
frame.shape = (1080, 1920, 3)
```

The model adapter may internally resize the frame to `352 × 352` or another model-specific input size.

## 2.2 Model Output

The model must return a foreground probability map aligned with the original frame.

### Output Specification

| Field | Type | Shape / Range | Description |
|---|---|---|---|
| `probability_map` | `np.ndarray` | `(H, W)` | Pixel-level foreground scores |
| `model_version` | string | — | Model version |
| `inference_time_ms` | float | ≥ 0 | Inference execution time |

Example:

```python
{
    "probability_map": np.ndarray,
    "model_version": "polyp-pvt-v1",
    "inference_time_ms": 42.5
}
```

The probability map must satisfy:

```python
probability_map.shape == (H, W)
probability_map.dtype == np.float32

np.all(np.isfinite(probability_map))
np.min(probability_map) >= 0.0
np.max(probability_map) <= 1.0
```

**Important:** These values represent model foreground scores, not calibrated clinical probabilities.

## 2.3 Required Inference Interface

The model team must expose a common interface for U-Net, Polyp-PVT, and Ensemble configurations.

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class PredictionResult:
    probability_map: np.ndarray
    model_version: str
    inference_time_ms: float


def predict(
    frame: np.ndarray,
    model_version: str
) -> PredictionResult:
    """
    Input:
        frame:
            RGB image
            shape (H, W, 3)
            dtype uint8

        model_version:
            Selected model configuration

    Output:
        probability_map:
            shape (H, W)
            dtype float32
            values in [0, 1]

        model_version:
            string

        inference_time_ms:
            float
    """
    ...
```

The model adapter must handle:

- Model-specific preprocessing.
- Model loading.
- Input resizing.
- Normalization.
- Output probability transformation.
- Resizing to original frame dimensions.
- Optional prediction-level fusion.

The backend must not depend on model-specific internal architectures.

## 2.4 Model Postprocessing

The backend receives the probability map and generates the binary mask.

### Input

```python
probability_map: np.ndarray
segmentation_threshold: float
min_region_area_ratio: float
```

### Output

```python
{
    "binary_mask": np.ndarray,
    "is_candidate": bool,
    "max_region_area_ratio": float,
    "contours": list
}
```

Binary mask generation:

```python
binary_mask = (
    probability_map >= segmentation_threshold
).astype(np.uint8)
```

Candidate-frame decision:

```text
1. Generate binary mask.
2. Find connected foreground components.
3. Calculate each component's area.
4. Select the largest component.
5. Divide its area by the original frame area.
6. Compare with the configured region-area threshold.
7. Return is_candidate = true or false.
```

The threshold values must be selected on validation data and saved in the analysis configuration.

### Model Team Deliverables

| Deliverable | Format |
|---|---|
| Model checkpoint | `.pth` / `.pt` |
| Inference code | `.py` |
| Model configuration | `.json` / `.yaml` |
| Preprocessing specification | JSON / YAML |
| Example RGB frame | `.png` / `.jpg` |
| Example probability map | `.npy` |
| Example binary mask | `.png` |
| Validation results | `.csv` / `.json` |
| Dependencies | `requirements.txt` |

---

# 3. Frontend → Backend: Video Upload

## 3.1 Upload Video

**Endpoint**

```http
POST /api/v1/videos
Content-Type: multipart/form-data
```

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Recorded colonoscopy video |

Example:

```typescript
const formData = new FormData();

formData.append("file", selectedFile);

const response = await api.post(
  "/videos",
  formData
);
```

### Output

```json
{
  "video_id": "video-001",
  "filename": "colonoscopy_demo.mp4",
  "video_url": "/api/v1/videos/video-001/stream",
  "duration_seconds": 750.0,
  "width": 1920,
  "height": 1080,
  "fps": 30.0,
  "total_frames": 22500
}
```

### Frontend Usage

The frontend uses:

- `video_id` to create an analysis task.
- `video_url` to load the video player.
- `duration_seconds` to render the timeline.
- `width` and `height` to calculate overlay scaling.

The backend must validate file type, file size, decoding support, and duration limits.

The streaming endpoint must support browser seeking, including HTTP Range Requests where applicable.

---

# 4. Frontend → Backend: Start Analysis

## 4.1 Create Analysis Task

**Endpoint**

```http
POST /api/v1/analyses
Content-Type: application/json
```

### Input

```json
{
  "video_id": "video-001",
  "sampling_mode": "interval",
  "interval_seconds": 1.0,
  "model_version": "polyp-pvt-v1"
}
```

### Input Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `video_id` | string | Yes | Uploaded video ID |
| `sampling_mode` | enum | Yes | `every_frame` or `interval` |
| `interval_seconds` | number / null | Conditional | Required for interval mode |
| `model_version` | string | Yes | Selected model configuration |

The backend retrieves the frozen segmentation threshold and region-area threshold associated with the selected model configuration.

### Output

```json
{
  "task_id": "task-001",
  "video_id": "video-001",
  "status": "queued",
  "configuration": {
    "sampling_mode": "interval",
    "interval_seconds": 1.0,
    "model_version": "polyp-pvt-v1",
    "segmentation_threshold": 0.5,
    "min_region_area_ratio": 0.01
  },
  "created_at": "2026-10-21T01:00:00Z"
}
```

### Backend Processing

After receiving the request:

```text
1. Validate video_id.
2. Validate sampling configuration.
3. Create analysis task.
4. Save immutable configuration snapshot.
5. Add task to processing queue.
6. Return task_id.
7. Execute video analysis asynchronously.
```

Do not run the complete video inference synchronously inside the HTTP request.

---

# 5. Backend → Frontend: Analysis Progress

## 5.1 Get Analysis Status

**Endpoint**

```http
GET /api/v1/analyses/{task_id}
```

### Input

| Parameter | Type | Description |
|---|---|---|
| `task_id` | string | Analysis task identifier |

### Output

```json
{
  "task_id": "task-001",
  "video_id": "video-001",
  "status": "running",
  "progress": {
    "target_frames": 750,
    "processed_frames": 300,
    "candidate_frames": 18,
    "failed_frames": 0,
    "percent": 40.0
  },
  "performance": {
    "elapsed_seconds": 52.5,
    "processing_fps": 5.71
  },
  "warnings": []
}
```

### Status Values

```typescript
type AnalysisStatus =
  | "queued"
  | "running"
  | "completed"
  | "cancelled"
  | "failed";
```

### Frontend Usage

The frontend displays:

- Analysis status.
- Progress percentage.
- Successfully analyzed frames.
- Candidate frames.
- Failed frames.
- Processing speed.
- Elapsed processing time.

### Polling

The frontend initially uses periodic polling:

```typescript
// Example: every 1500 ms
refetchInterval: 1500
```

Polling should stop when the task reaches a terminal state:

- `completed`
- `cancelled`
- `failed`

The frontend must not interpret processing speed as original video playback FPS.

---

# 6. Backend → Frontend: Frame Results

This is the most important API for frontend visualization.

## 6.1 Get Frame Results

**Endpoint**

```http
GET /api/v1/analyses/{task_id}/frames
```

### Input

| Parameter | Type | Required | Description |
|---|---|---|---|
| `task_id` | string | Yes | Analysis task |
| `after` | string | No | Pagination cursor |
| `limit` | integer | No | Maximum returned records |
| `candidate_only` | boolean | No | Filter candidate frames |

Example:

```http
GET /api/v1/analyses/task-001/frames?limit=100&candidate_only=true
```

### Output

```json
{
  "task_id": "task-001",
  "items": [
    {
      "frame_id": "frame-842",
      "frame_index": 842,
      "timestamp_seconds": 252.0,
      "width": 1920,
      "height": 1080,
      "status": "completed",
      "is_candidate": true,
      "max_region_area_ratio": 0.037,
      "source_frame_url": "/api/v1/assets/frames/frame-842.jpg",
      "thumbnail_url": "/api/v1/assets/thumbs/frame-842.jpg",
      "mask_url": "/api/v1/assets/masks/frame-842.png",
      "contours": [
        [
          [850, 390],
          [900, 375],
          [970, 400],
          [990, 470]
        ]
      ],
      "review_status": "unreviewed"
    }
  ],
  "next_cursor": null
}
```

### Required Frame Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `frame_id` | string | Yes | Unique frame result ID |
| `frame_index` | integer | Yes | Original decoded frame index |
| `timestamp_seconds` | number | Yes | Actual source timestamp |
| `width` | integer | Yes | Original frame width |
| `height` | integer | Yes | Original frame height |
| `status` | enum | Yes | Processing result |
| `is_candidate` | boolean / null | Yes | Candidate decision |
| `max_region_area_ratio` | number / null | Yes | Largest region area ratio |
| `source_frame_url` | string / null | Yes | Exact extracted frame |
| `thumbnail_url` | string / null | Yes | Preview image |
| `mask_url` | string / null | Yes | Binary mask |
| `contours` | array | Yes | Polygon coordinates |
| `review_status` | enum | Yes | Human review status |

For failed frames, `is_candidate` and area ratio must be `null`, not `false` or zero.

### Coordinate Convention

All contour coordinates use the original decoded frame resolution.

```text
Origin: top-left
X axis: left → right
Y axis: top → bottom

Coordinate:
[x, y]
```

The frontend is responsible for scaling coordinates to the displayed video size, accounting for aspect ratio and letterboxing.

### Frontend Usage

| Component | Fields |
|---|---|
| DetectionOverlay | `timestamp_seconds`, `contours`, `mask_url` |
| DetectionPanel | `is_candidate`, `max_region_area_ratio` |
| VideoTimeline | `timestamp_seconds`, `is_candidate` |
| KeyFrames | `thumbnail_url`, `timestamp_seconds` |
| PolypDetail | `source_frame_url`, `mask_url`, `frame_index` |

**Important:** The number of candidate frames must not be presented as the number of distinct physical polyps.

---

# 7. Backend → Frontend: Keyframes

## 7.1 Get Keyframes

**Endpoint**

```http
GET /api/v1/analyses/{task_id}/keyframes
```

### Input

```text
task_id: string
```

### Output

```json
{
  "task_id": "task-001",
  "keyframes": [
    {
      "frame_id": "frame-842",
      "timestamp_seconds": 252.0,
      "thumbnail_url": "/api/v1/assets/thumbs/frame-842.jpg",
      "is_candidate": true
    },
    {
      "frame_id": "frame-950",
      "timestamp_seconds": 285.0,
      "thumbnail_url": "/api/v1/assets/thumbs/frame-950.jpg",
      "is_candidate": false
    }
  ]
}
```

### Frontend Behavior

When a user clicks a keyframe:

1. Select the corresponding `frame_id`.
2. Seek the video to `timestamp_seconds`.
3. Load the exact extracted frame for inspection.
4. Display the corresponding segmentation mask.
5. Update the selected state in the detection panel and timeline.

Browser video seeking may not reproduce the exact decoded inference frame. The saved `source_frame_url` must be used for precise mask review.

---

# 8. Frontend → Backend: Mask Editing

## 8.1 Save Edited Mask

**Endpoint**

```http
PUT /api/v1/frames/{frame_id}/review
Content-Type: multipart/form-data
```

### Input

| Field | Type | Required | Description |
|---|---|---|---|
| `frame_id` | string | Yes | Selected frame |
| `base_revision_id` | string / null | Yes | Revision being edited |
| `edited_mask` | PNG File | Yes | Edited binary mask |
| `review_status` | string | Yes | Review status |
| `notes` | string | No | User comments |

### Example

```typescript
const formData = new FormData();

formData.append("edited_mask", maskFile);
formData.append("review_status", "reviewed");
formData.append("notes", "Boundary refined");

if (baseRevisionId) {
  formData.append("base_revision_id", baseRevisionId);
}
```

### Output

```json
{
  "frame_id": "frame-842",
  "revision_id": "rev-002",
  "review_status": "reviewed",
  "edited_mask_url": "/api/v1/assets/reviews/rev-002.png",
  "notes": "Boundary refined",
  "saved_at": "2026-10-21T01:20:00Z"
}
```

### Requirements

- The original AI mask must remain unchanged.
- Edited masks must be stored separately.
- Each successful save creates a new revision.
- The backend must detect revision conflicts.
- The frontend should support Brush, Eraser, Undo, Redo, and Reset.
- Editing one frame must not automatically change adjacent frames.

---

# 9. Backend → Frontend: Reports

## 9.1 Get Analysis Report

**Endpoint**

```http
GET /api/v1/analyses/{task_id}/report
```

### Input

```text
task_id: string
```

### Output

```json
{
  "task_id": "task-001",
  "video_id": "video-001",
  "status": "completed",
  "model_version": "polyp-pvt-v1",
  "sampling": {
    "mode": "interval",
    "interval_seconds": 1.0
  },
  "summary": {
    "analyzed_frames": 750,
    "candidate_frames": 42,
    "failed_frames": 0,
    "candidate_frame_ratio": 0.056,
    "reviewed_frames": 15,
    "processing_time_seconds": 120.5
  },
  "generated_at": "2026-10-21T01:30:00Z"
}
```

### Report Requirements

The report must contain:

- Video information.
- Model version.
- Sampling configuration.
- Analysis completion status.
- Successfully analyzed frame count.
- Candidate frame count and ratio.
- Failed frame count.
- Processing time.
- Candidate-frame temporal distribution.
- Segmented area statistics.
- Representative frame images and overlays.
- Human review records.

Reports must distinguish automatic predictions from manually edited results.

### 9.2 Export Report

**Endpoint**

```http
POST /api/v1/analyses/{task_id}/exports
```

### Input

```json
{
  "format": "csv",
  "include_reviewed_masks": true
}
```

### Output

```json
{
  "export_id": "export-001",
  "status": "completed",
  "download_url": "/api/v1/exports/export-001/download"
}
```

Supported formats should be confirmed by the backend team.

Suggested formats:

- Printable HTML / PDF.
- Frame-level CSV.
- Selected binary masks.
- Selected segmentation overlays.

---

# 10. Frontend File Organization

The frontend must strictly preserve the existing structure.

```text
SystemCode/
├── backend/
└── frontend/
    └── src/
        ├── assets/
        ├── components/
        ├── pages/
        │   ├── AnalysisPage.tsx
        │   └── ReportsPage.tsx
        └── utils/
            ├── dataQueries.ts
            └── endpoints.ts
```

## 10.1 dataQueries.ts

**Purpose:** Store mock data and mock query functions.

Input:

```typescript
getMockFrameResults(taskId: string)
```

Output:

```typescript
Promise<FrameResultsResponse>
```

Requirements:

- Mock data must follow the same schema as the backend API.
- Mock results must be deterministic.
- Do not hardcode mock data inside UI components.
- Do not fabricate real clinical inference results.

## 10.2 endpoints.ts

**Purpose:** Centralize all real FastAPI requests.

Example:

```typescript
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 15000,
});

export async function getFrameResults(
  taskId: string
) {
  const response = await api.get(
    `/analyses/${taskId}/frames`
  );

  return response.data;
}
```

All backend requests must be placed in `endpoints.ts`.

The frontend must not call backend APIs directly inside reusable UI components.

---

# 11. Data Consistency Requirements

All teams must agree on the following conventions.

| Item | Convention |
|---|---|
| Time unit | Seconds |
| Frame index | Original decoded source-frame index |
| Image format | RGB |
| Probability map | float32, values 0–1 |
| Binary mask | Single-channel 0/1 internally; PNG 0/255 for export |
| Mask dimensions | Original frame resolution |
| Contour coordinates | Original frame pixel coordinates |
| Candidate count | Number of successfully analyzed candidate frames |
| Failed frames | Excluded from candidate and negative counts |
| Unsampled frames | Not treated as negative |
| Review revisions | Separate from original AI predictions |
| Model version | Stored with each analysis task |

The frontend must not independently recalculate authoritative analysis statistics.

Backend summary statistics and exported report statistics must use the same saved frame records.

---

# 12. Integration Testing Checklist

## 12.1 Model → Backend

- [ ] Model checkpoint loads successfully.
- [ ] Model accepts RGB frames.
- [ ] Model returns probability maps in the agreed format.
- [ ] Probability maps align with original frame dimensions.
- [ ] Thresholding produces correct binary masks.
- [ ] Candidate-frame rules are implemented consistently.
- [ ] Model version and preprocessing settings are recorded.

## 12.2 Frontend → Backend

- [ ] Video upload succeeds.
- [ ] Video metadata is returned correctly.
- [ ] Video playback and seeking work.
- [ ] Analysis task creation succeeds.
- [ ] Progress updates are available through polling.
- [ ] Frame results can be retrieved incrementally.
- [ ] Candidate-frame counts match backend records.
- [ ] Timeline markers use correct timestamps.
- [ ] Keyframe selection navigates to the correct frame.
- [ ] Canvas overlays align with original frames.
- [ ] Mask edits can be saved and reopened.
- [ ] Original masks remain unchanged.
- [ ] Reports match saved analysis results.

## 12.3 Error and Edge Cases

- [ ] Invalid video format.
- [ ] Unsupported codec.
- [ ] Oversized video.
- [ ] Video decoding failure.
- [ ] Model inference failure.
- [ ] No candidate frames.
- [ ] Cancelled analysis.
- [ ] Partially completed analysis.
- [ ] Missing mask or thumbnail.
- [ ] Revision conflict.
- [ ] Page refresh during processing.
- [ ] Duplicate frame-result writes.

---

# 13. Open Questions for Technical Alignment

The following decisions must be confirmed before integration.

| Question | Responsible Team | Status |
|---|---|---|
| Which model version will be deployed initially? | Model | Pending |
| What is the final model input size? | Model | Pending |
| What preprocessing and normalization are required? | Model | Pending |
| What segmentation threshold will be used? | Model | Pending |
| What minimum region-area ratio will be used? | Model + Backend | Pending |
| Which video formats and codecs are supported? | Backend | Pending |
| What are the maximum upload size and video duration? | Backend | Pending |
| Which frame sampling intervals are supported? | Backend | Pending |
| How are keyframes selected? | Backend + Frontend | Pending |
| Should the frontend use contours, binary masks, or both? | Backend + Frontend | Pending |
| What polling interval should be used? | Backend + Frontend | Pending |
| How will report export files be generated and stored? | Backend | Pending |

---

# 14. Required Deliverables Before Integration

## Model Team

- [ ] Model checkpoint.
- [ ] Unified inference interface.
- [ ] Preprocessing configuration.
- [ ] Model version information.
- [ ] Sample probability maps.
- [ ] Sample binary masks.
- [ ] Validation metrics.
- [ ] Environment setup instructions.

## Backend Team

- [ ] Video upload API.
- [ ] Video metadata and playback API.
- [ ] Analysis task creation API.
- [ ] Analysis progress API.
- [ ] Frame results API.
- [ ] Keyframes API.
- [ ] Mask and thumbnail storage.
- [ ] Review revision API.
- [ ] Report API.
- [ ] Export API.
- [ ] OpenAPI documentation and example responses.

## Frontend Team

- [ ] AnalysisPage.
- [ ] ReportsPage.
- [ ] VideoPlayer.
- [ ] DetectionOverlay.
- [ ] DetectionPanel.
- [ ] VideoTimeline.
- [ ] KeyFrames.
- [ ] Mask Editor.
- [ ] Mock data in `dataQueries.ts`.
- [ ] API functions in `endpoints.ts`.
- [ ] Integration testing.

---

# 15. Definition of Done

The system integration is complete when:

1. A user can upload a valid colonoscopy video.
2. The backend can create and execute an analysis task.
3. The selected model generates probability maps for sampled frames.
4. The backend converts model outputs into binary masks and candidate-frame records.
5. The frontend can retrieve processing progress and frame results.
6. The video player, segmentation overlays, timeline, and keyframes are synchronized.
7. Users can inspect and edit selected segmentation masks.
8. Edited masks are saved separately from original model predictions.
9. Reports are generated from consistent saved backend records.
10. The system handles cancellation, failures, partial results, and page refreshes.

**Final Integration Principle:**

The model provides standardized segmentation predictions. The backend provides consistent, timestamped, persistent analysis records. The frontend consumes those records through stable APIs without depending on model-specific implementation details.

All three teams must agree on the data contracts before final integration.