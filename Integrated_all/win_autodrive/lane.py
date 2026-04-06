from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .messages import Detection, LaneInfo, TargetPoint


@dataclass
class BirdEyeConfig:
    """BirdEyeConfig: 원근변환(perspective transform) 파라미터.

    원근변환(perspective transform): 4개 점(src)을 다른 4개 점(dst)으로 매핑하는
    3x3 변환 행렬을 구해 영상의 시점을 바꾸는 변환.
    예시: 도로를 위에서 내려다본 것처럼 펴서 차선/경계가 더 단순한 모양이 되게 함.
    """

    src: np.ndarray  # shape (4,2) float32
    dst: np.ndarray  # shape (4,2) float32
    cut_row: int = 300


def detections_to_edge_image(
    detections: List[Detection],
    height: int,
    width: int,
    lane_class_name: str = "lane2",
    thickness: int = 2,
) -> np.ndarray:
    """Convert lane mask detections into a binary edge image.

    - edge(에지): 밝기가 급격히 변하는 경계(예: 차선 페인트의 경계).
      예시: 흰 선의 테두리 부분.

    이 함수는 YOLO 세그멘테이션 mask polygon을 폴리라인으로 그려
    (0/255) 흑백 영상으로 만든다.
    """

    edge = np.zeros((height, width), dtype=np.uint8)

    for det in detections:
        if det.class_name != lane_class_name:
            continue
        if det.mask is None or not det.mask.polygon:
            continue
        pts = np.array([[p.x, p.y] for p in det.mask.polygon], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(edge, [pts], isClosed=True, color=255, thickness=thickness)

    return edge


def bird_eye_warp(edge_img: np.ndarray, cfg: BirdEyeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bird-eye warp. Returns (warped, M).

    M(변환행렬): 3x3 행렬. cv2.warpPerspective로 적용.
    """

    M = cv2.getPerspectiveTransform(cfg.src.astype(np.float32), cfg.dst.astype(np.float32))
    h, w = edge_img.shape[:2]
    warped = cv2.warpPerspective(edge_img, M, (w, h))

    # ROI: 상단 잘라내기 (노이즈 많은 구간 제거)
    if 0 < cfg.cut_row < h:
        warped[: cfg.cut_row, :] = 0

    return warped, M


def roi_rectangle_below(img_gray: np.ndarray, y: int, x_min: int, x_max: int) -> np.ndarray:
    """Keep only the region below y (and between x_min..x_max)."""

    out = np.zeros_like(img_gray)
    h, w = img_gray.shape[:2]
    y = int(np.clip(y, 0, h))
    x_min = int(np.clip(x_min, 0, w - 1))
    x_max = int(np.clip(x_max, 0, w))
    out[y:h, x_min:x_max] = img_gray[y:h, x_min:x_max]
    return out


def dominant_gradient(edge_img: np.ndarray, theta_limit_deg: float = 15.0) -> float:
    """Estimate dominant lane direction as slope using HoughLines.

    Hough transform(허프 변환): 에지 픽셀들을 직선 후보들로 누적해서
    가장 강한 직선을 찾는 방법.
    예시: 차선 에지 이미지에서 전체적으로 가장 '긴' 직선 방향을 찾음.

    Returns slope in image coordinates (dy/dx). If no line, returns 0.
    """

    lines = cv2.HoughLines(edge_img, 1, np.pi / 180.0, 200)
    if lines is None:
        return 0.0

    theta_limit = math.radians(theta_limit_deg)
    selected = [ln for ln in lines if abs(ln[0][1] - math.pi / 2) <= theta_limit]
    if not selected:
        return 0.0

    theta_values = [ln[0][1] for ln in selected]
    theta_avg = float(np.mean(theta_values))

    # theta is angle of normal; line direction angle = theta - pi/2
    line_angle = theta_avg - math.pi / 2.0
    # slope = tan(direction angle)
    slope = math.tan(line_angle)
    return float(slope)


def lane_center_x(edge_img: np.ndarray, y: int, thickness: int = 100) -> Optional[int]:
    """Find lane center x at row y by scanning white pixels.

    Works best on bird-eye+ROI images where lane edges appear as two blobs.
    Returns None if insufficient data.
    """

    h, w = edge_img.shape
    y = int(np.clip(y, 0, h - 1))

    x_min = max(0, int(w * 0.05))
    x_max = min(w, int(w * 0.95))

    roi = edge_img[max(0, y - thickness): min(h, y + thickness), x_min:x_max]
    xs = np.where(roi == 255)[1]
    if xs.size < 50:
        return None

    left = int(xs.min() + x_min)
    right = int(xs.max() + x_min)
    return int((left + right) / 2)


from .profiler import profile

@profile("LaneExtraction")
def extract_lane_info(
    detections: List[Detection],
    frame_h: int,
    frame_w: int,
    bird_cfg: BirdEyeConfig,
    lane_class_name: str = "lane2",
    edge_thickness: int = 2,
    target_rows: Sequence[int] = (0, 100, 150, 200),
) -> Tuple[np.ndarray, np.ndarray, Optional[LaneInfo]]:
    """Full lane pipeline:

    1) YOLO mask -> edge image (original)
    2) bird-eye warp
    3) ROI crop below cut_row
    4) dominant slope + center target points

    Returns (edge_original, edge_bird, LaneInfo or None)
    """

    edge = detections_to_edge_image(
        detections=detections,
        height=frame_h,
        width=frame_w,
        lane_class_name=lane_class_name,
        thickness=edge_thickness,
    )

    edge_bird, _M = bird_eye_warp(edge, bird_cfg)

    # Also limit horizontal span similar to ROS version
    edge_bird_roi = roi_rectangle_below(
        edge_bird,
        y=bird_cfg.cut_row,
        x_min=int(frame_w * 0.15),
        x_max=int(frame_w * 0.85),
    )

    slope = dominant_gradient(edge_bird_roi, theta_limit_deg=15.0)

    tps: List[TargetPoint] = []
    for dy in target_rows:
        y = bird_cfg.cut_row + int(dy)
        x = lane_center_x(edge_bird_roi, y=y, thickness=100)
        if x is None:
            continue
        tps.append(TargetPoint(x=int(x), y=int(y)))

    if not tps:
        return edge, edge_bird_roi, None

    return edge, edge_bird_roi, LaneInfo(slope=float(slope), target_points=tps)
