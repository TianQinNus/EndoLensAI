"""Backend helper. Region-area gate is an UNVALIDATED engineering setting."""
from dataclasses import dataclass
import numpy as np


@dataclass
class PostprocessResult:
    binary_mask: np.ndarray
    is_candidate: bool
    max_region_area_ratio: float
    contours: list
    regions: list
    foreground_score: float | None
    segmented_area_ratio: float
    component_count: int
    display_geometry_truncated: bool


def postprocess(probability_map, segmentation_threshold=.5, min_region_area_ratio=0.):
    import cv2
    p = probability_map
    if (not isinstance(p, np.ndarray) or p.dtype != np.float32 or p.ndim != 2 or p.size == 0
            or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1):
        raise ValueError("Require finite float32(H,W) probability in[0,1]")
    if not np.isfinite(segmentation_threshold) or not 0 < segmentation_threshold < 1 or not np.isfinite(min_region_area_ratio) or not 0 <= min_region_area_ratio <= 1:
        raise ValueError("Invalid postprocessing thresholds")
    mask = np.ascontiguousarray((p >= segmentation_threshold).astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sums = np.bincount(labels.reshape(-1), weights=p.reshape(-1), minlength=count)
    order = sorted(range(1, count), key=lambda k: (-int(stats[k,4]), k))
    regions = []
    for label in order[:100]:
        x, y, width, height, area = map(int, stats[label])
        regions.append({"component_id": label, "area_pixels": area, "area_ratio": area/p.size,
                        "bbox_xywh": [x,y,width,height], "foreground_score": float(sums[label]/area)})
    regions.sort(key=lambda r: (-r["area_pixels"], r["component_id"]))
    # All original foreground is retained. Area gate affects candidate status ONLY.
    raw, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    raw = sorted(raw, key=cv2.contourArea, reverse=True)
    truncated = len(raw) > 100 or count-1 > 100
    for contour in raw[:100]:
        simplified = cv2.approxPolyDP(contour, 1., True).reshape(-1, 2)
        if len(simplified) > 256:
            simplified = simplified[np.linspace(0, len(simplified)-1, 256, dtype=int)]
            truncated = True
        if len(simplified) >= 3:
            contours.append(simplified.astype(int).tolist())
    largest = regions[0]["area_ratio"] if regions else 0.
    return PostprocessResult(mask, bool(regions and largest >= min_region_area_ratio), float(largest), contours,
                             regions, regions[0]["foreground_score"] if regions else None, float(mask.mean()), count-1, truncated)
