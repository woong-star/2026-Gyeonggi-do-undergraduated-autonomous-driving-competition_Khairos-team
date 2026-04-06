from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .messages import BoundingBox2D, Detection, Mask, Point2D


@dataclass
class YoloConfig:
    model_path: str = "best.pt"
    device: str = "cpu"
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 100
    imgsz: int = 640


from .profiler import profile

class YoloV8:
    """YOLOv8 wrapper (Ultralytics).

    - Ultralytics: YOLOv8 모델/추론 라이브러리 (예: lane/traffic_light 세그멘테이션).
      예시: "best.pt"를 학습한 뒤 여기서 로드해서 프레임마다 예측.
    """

    def __init__(self, cfg: YoloConfig):
        self.cfg = cfg
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "ultralytics 미설치 또는 import 실패. 'pip install ultralytics' 후 재실행하세요."
            ) from e

        self.model = YOLO(cfg.model_path)
        # Force move to device
        if "cuda" in cfg.device and torch.cuda.is_available():
            self.model.to("cuda")
        
        # names: dict[int,str]
        self.names = self.model.names if hasattr(self.model, "names") else {}

        # GPU 사용 여부 확인용 출력
        if "cuda" in cfg.device and torch.cuda.is_available():
             print(f"[YOLO] Loading model on GPU: {torch.cuda.get_device_name(0)}")
        else:
             print(f"[YOLO] Loading model on CPU (Is CUDA available? {torch.cuda.is_available()})")

    def class_names(self) -> List[str]:
        if isinstance(self.names, dict):
            return [self.names[k] for k in sorted(self.names.keys())]
        if isinstance(self.names, list):
            return list(self.names)
        return []

    @profile("YOLO")
    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Return detections for a single BGR frame."""
        # Runtime Check (First frame only)
        if not hasattr(self, "_device_checked"):
            try:
                # YOLOv8 model parameters device check
                param_device = next(self.model.model.parameters()).device
                print(f"\n[VERIFICATION] YOLO Model is on: {param_device} (Should be cuda:0)")
                self._device_checked = True
            except Exception:
                pass

        # Ultralytics returns a list-like of Results. We use first item.
        results = self.model(
            frame_bgr,
            device=self.cfg.device,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            max_det=self.cfg.max_det,
            imgsz=self.cfg.imgsz,
            half=True,
            verbose=False,
        )
        if results is None or len(results) == 0:
            return []

        r0 = results[0]
        h, w = frame_bgr.shape[:2]

        dets: List[Detection] = []

        boxes = getattr(r0, "boxes", None)
        masks = getattr(r0, "masks", None)

        n = 0
        if boxes is not None and hasattr(boxes, "cls"):
            n = len(boxes)

        # masks.xy is list of (N_i x 2) polygons in pixel coords
        masks_xy: Optional[List[np.ndarray]] = None
        if masks is not None and hasattr(masks, "xy") and masks.xy is not None:
            masks_xy = list(masks.xy)

        for i in range(n):
            cls_id = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], "item") else int(boxes.cls[i])
            score = float(boxes.conf[i].item()) if hasattr(boxes.conf[i], "item") else float(boxes.conf[i])
            name = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else str(cls_id)

            # xywh: [cx, cy, w, h]
            if hasattr(boxes, "xywh"):
                xywh = boxes.xywh[i]
                cx = float(xywh[0].item())
                cy = float(xywh[1].item())
                bw = float(xywh[2].item())
                bh = float(xywh[3].item())
                bbox = BoundingBox2D(cx=cx, cy=cy, w=bw, h=bh)
            else:
                bbox = None

            mask_obj = None
            if masks_xy is not None and i < len(masks_xy) and masks_xy[i] is not None:
                poly = masks_xy[i]
                if poly.size >= 6:  # at least 3 points
                    pts = [Point2D(float(x), float(y)) for x, y in poly]
                    mask_obj = Mask(polygon=pts, height=h, width=w)

            dets.append(
                Detection(
                    class_id=cls_id,
                    class_name=name,
                    score=score,
                    bbox=bbox,
                    mask=mask_obj,
                )
            )

        return dets
