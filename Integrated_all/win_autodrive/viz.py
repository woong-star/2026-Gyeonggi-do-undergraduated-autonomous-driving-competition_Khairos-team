from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .lane import BirdEyeConfig
from .messages import LaneInfo, MotionCommand, PathPlanningResult


@dataclass
class VizConfig:
    """VizConfig: 디버그 화면 표시 설정."""

    window_name: str = "view"
    scale: float = 0.5


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def draw_path_on_bird(edge_bird_roi: np.ndarray, path: Optional[PathPlanningResult]) -> np.ndarray:
    img = _to_bgr(edge_bird_roi)
    if path is None:
        return img
    for x, y in zip(path.x_points, path.y_points):
        cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)
    return img


def build_mosaic(frame_bgr: np.ndarray,
                 edge_original: np.ndarray,
                 edge_bird_roi: np.ndarray,
                 bird_cfg: BirdEyeConfig,
                 lane: Optional[LaneInfo],
                 path: Optional[PathPlanningResult],
                 cmd: Optional[MotionCommand]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    # TL: original frame + ROI polygon
    tl = frame_bgr.copy()
    pts = bird_cfg.src.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(tl, [pts], True, (0, 0, 255), 2)

    if lane is None:
        cv2.putText(tl, "Lane: NOT FOUND", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(tl, f"Lane slope={lane.slope:+.3f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # steering / command info
    if cmd is None:
        cv2.putText(tl, "Steer: -", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(
            tl,
            f"Steer cmd={cmd.steering:+d}  L={cmd.left_speed:+d}  R={cmd.right_speed:+d}",
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

    # TR: bird ROI edge
    tr = _to_bgr(edge_bird_roi)

    # BL: original edge
    bl = _to_bgr(edge_original)

    # BR: bird ROI edge + path points
    br = draw_path_on_bird(edge_bird_roi, path)

    # Resize all to half
    tl_s = cv2.resize(tl, (w // 2, h // 2))
    tr_s = cv2.resize(tr, (w // 2, h // 2))
    bl_s = cv2.resize(bl, (w // 2, h // 2))
    br_s = cv2.resize(br, (w // 2, h // 2))

    top = np.hstack([tl_s, tr_s])
    bot = np.hstack([bl_s, br_s])
    return np.vstack([top, bot])


class Visualizer:
    def __init__(self, cfg: VizConfig):
        self.cfg = cfg
        cv2.namedWindow(cfg.window_name, cv2.WINDOW_NORMAL)

    def show(self,
             frame_bgr: np.ndarray,
             edge_original: np.ndarray,
             edge_bird_roi: np.ndarray,
             bird_cfg: BirdEyeConfig,
             lane: Optional[LaneInfo],
             path: Optional[PathPlanningResult],
             cmd: Optional[MotionCommand] = None) -> int:
        mosaic = build_mosaic(frame_bgr, edge_original, edge_bird_roi, bird_cfg, lane, path, cmd)
        cv2.imshow(self.cfg.window_name, mosaic)
        return int(cv2.waitKey(1) & 0xFF)

    def show_simple(self,
                    frame_bgr: np.ndarray,
                    lane: Optional[LaneInfo],
                    path: Optional[PathPlanningResult],
                    cmd: Optional[MotionCommand]) -> int:
        """단순화된 시각화: 원본 영상에 오버레이만 해서 출력 (Mosaic 생략)."""
        img = frame_bgr.copy()
        h, w = img.shape[:2]

        # Draw Lane Text
        if lane is None:
            cv2.putText(img, "Lane: NOT FOUND", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(img, f"Slope={lane.slope:+.3f}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            # Draw target points
            for tp in lane.target_points:
                # bird-eye좌표를 원본 좌표로 역변환하는 건 복잡하므로, 여기서는 간략히 텍스트나 다른 정보만 표시
                # 혹은 bird-eye 상의 점을 표시하고 싶다면 역변환 행렬이 필요함.
                # 편의상 여기서는 생략하고, 텍스트 정보 위주로 표시.
                pass

        # Draw Path/Cmd Info
        if cmd:
            cv2.putText(img, f"Steer={cmd.steering:+d} V={cmd.left_speed}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow(self.cfg.window_name, img)
        return int(cv2.waitKey(1) & 0xFF)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.cfg.window_name)
        except Exception:
            pass
