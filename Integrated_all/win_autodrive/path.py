from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .messages import LaneInfo, PathPlanningResult


@dataclass
class PathConfig:
    """PathConfig: 경로 생성 파라미터.

    CubicSpline(3차 스플라인): 여러 점을 부드럽게 연결하는
    3차 다항식 조각(piecewise polynomial) 보간.
    예시: y에 따라 x가 부드럽게 변하는 경로를 생성.
    """

    sample_count: int = 40


def _try_cubic_spline(x: np.ndarray, y: np.ndarray, sample_count: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    try:
        from scipy.interpolate import CubicSpline
    except Exception:
        return x, y, False

    # Parameter t along points (0..1)
    t = np.linspace(0.0, 1.0, len(x))
    csx = CubicSpline(t, x)
    csy = CubicSpline(t, y)

    ts = np.linspace(0.0, 1.0, sample_count)
    xs = csx(ts)
    ys = csy(ts)
    return xs, ys, True


def plan_path(lane: LaneInfo, cfg: PathConfig, car_center: Tuple[float, float]) -> PathPlanningResult:
    """Generate a smooth path from lane target points.

    car_center: (x,y) in same coordinate system as target points (pixel).
    """

    pts = [(float(tp.x), float(tp.y)) for tp in lane.target_points]
    pts.append((float(car_center[0]), float(car_center[1])))

    # sort by y (ascending) to have consistent direction
    pts = sorted(pts, key=lambda p: p[1])

    x = np.array([p[0] for p in pts], dtype=np.float64)
    y = np.array([p[1] for p in pts], dtype=np.float64)

    if len(pts) >= 4:
        xs, ys, ok = _try_cubic_spline(x, y, cfg.sample_count)
        if ok:
            return PathPlanningResult(x_points=[float(v) for v in xs], y_points=[float(v) for v in ys])

    # Fallback: linear interpolation by y
    # Create monotonic axis using index parameter
    t = np.linspace(0.0, 1.0, len(x))
    ts = np.linspace(0.0, 1.0, cfg.sample_count)
    xs = np.interp(ts, t, x)
    ys = np.interp(ts, t, y)

    return PathPlanningResult(x_points=[float(v) for v in xs], y_points=[float(v) for v in ys])
