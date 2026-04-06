from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Point2D:
    """2D point in pixel coordinates."""

    x: float
    y: float


@dataclass
class BoundingBox2D:
    """Axis-aligned bounding box using center+size (pixel units)."""

    cx: float
    cy: float
    w: float
    h: float

    def xyxy_int(self) -> Tuple[int, int, int, int]:
        x_min = int(round(self.cx - self.w / 2.0))
        y_min = int(round(self.cy - self.h / 2.0))
        x_max = int(round(self.cx + self.w / 2.0))
        y_max = int(round(self.cy + self.h / 2.0))
        return x_min, y_min, x_max, y_max


@dataclass
class Mask:
    """Polygon mask (list of vertices) + original image size."""

    polygon: List[Point2D] = field(default_factory=list)
    height: int = 0
    width: int = 0


@dataclass
class Detection:
    """Single detection result."""

    class_id: int
    class_name: str
    score: float
    bbox: Optional[BoundingBox2D] = None
    mask: Optional[Mask] = None


@dataclass
class TargetPoint:
    """TargetPoint: point on the lane center in bird/ROI image (pixel units)."""

    x: int
    y: int


@dataclass
class LaneInfo:
    """LaneInfo: lane slope + multiple lookahead target points."""

    slope: float
    target_points: List[TargetPoint] = field(default_factory=list)


@dataclass
class PathPlanningResult:
    """Planned path as discrete points (pixel units)."""

    x_points: List[float] = field(default_factory=list)
    y_points: List[float] = field(default_factory=list)


@dataclass
class MotionCommand:
    """MotionCommand: steering + left/right speed command."""

    steering: int
    left_speed: int
    right_speed: int
