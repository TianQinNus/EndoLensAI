"""The released rule deletes low-mean-score regions from mask and geometry."""
from dataclasses import dataclass
import numpy as np
import cv2
from .constants import PIXEL_THRESHOLD, REGION_MEAN_THRESHOLD, POSTPROCESS_VERSION

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
    raw_component_count: int
    removed_component_count: int
    postprocessing_version: str = POSTPROCESS_VERSION

def postprocess(probability_map,segmentation_threshold=PIXEL_THRESHOLD,
                min_region_area_ratio=0.,min_region_mean_score=REGION_MEAN_THRESHOLD):
    p=probability_map
    if (not isinstance(p,np.ndarray) or p.dtype!=np.float32 or p.ndim!=2 or p.size==0
        or not np.isfinite(p).all() or p.min()<0 or p.max()>1):
        raise ValueError('Require finite native-resolution float32(H,W) probabilities in [0,1]')
    if segmentation_threshold!=.5 or min_region_area_ratio!=0. or min_region_mean_score!=.8:
        raise ValueError('This release fixes pixel=0.5, component mean>=0.8, area gate=0; use a new version for a different rule')
    raw=np.ascontiguousarray((p>=.5).astype(np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(raw,connectivity=8,ltype=cv2.CV_32S)
    sums=np.bincount(labels.ravel(),weights=p.ravel(),minlength=count)
    keep=np.zeros(count,dtype=np.uint8);all_regions=[]
    for label in range(1,count):
        x,y,w,h,area=map(int,stats[label]);score=float(sums[label]/area)
        # Compare the unrounded native-region mean, never the max pixel or UI text.
        if score>=.8:
            keep[label]=1
            all_regions.append({'component_id':label,'area_pixels':area,'area_ratio':area/p.size,
                'bbox_xywh':[x,y,w,h],'foreground_score':score})
    mask=np.ascontiguousarray(keep[labels],dtype=np.uint8)
    all_regions.sort(key=lambda r:(-r['area_pixels'],r['component_id']))
    contours=[];raw_contours,_=cv2.findContours(mask.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    truncated=len(raw_contours)>100 or len(all_regions)>100
    for contour in sorted(raw_contours,key=cv2.contourArea,reverse=True)[:100]:
        points=cv2.approxPolyDP(contour,1.,True).reshape(-1,2)
        if len(points)>256:
            points=points[np.linspace(0,len(points)-1,256,dtype=int)];truncated=True
        if len(points)>=3:contours.append(points.astype(int).tolist())
    largest=all_regions[0] if all_regions else None
    return PostprocessResult(mask,bool(mask.any()),largest['area_ratio'] if largest else 0.,contours,
        all_regions[:100],largest['foreground_score'] if largest else None,float(mask.mean()),len(all_regions),
        truncated,count-1,count-1-len(all_regions))
