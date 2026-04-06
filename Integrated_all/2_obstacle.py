from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from win_autodrive.lane import BirdEyeConfig, extract_lane_info, lane_center_x
from win_autodrive.messages import MotionCommand
from win_autodrive.motion import MotionConfig, plan_motion
from win_autodrive.path import PathConfig, plan_path
from win_autodrive.serial_io import SerialConfig, SerialSender
from win_autodrive.viz import VizConfig, Visualizer
from win_autodrive.yolo import YoloConfig, YoloV8
from win_autodrive.profiler import profile, print_stats


LANE_CLASSES = ("lane1", "lane2")
OBJ_CLASSES = ("traffic_light", "car_rear")

TL_WIN_NAME = "traffic_light"


def _filter_by_class(detections, allowed: Tuple[str, ...]):
    return [d for d in detections if d.class_name in allowed]


def _open_cam(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    return cap


@dataclass
class Stream:
    cap: cv2.VideoCapture
    is_video: bool
    path: Optional[str] = None


def _open_stream(cam_index: int, video_path: Optional[str], width: int, height: int, fps: int) -> Stream:
    if video_path:
        cap = cv2.VideoCapture(video_path)
        return Stream(cap=cap, is_video=True, path=video_path)
    cap = _open_cam(cam_index, width, height, fps)
    return Stream(cap=cap, is_video=False, path=None)


def _read_stream(stream: Stream, loop: bool) -> Tuple[bool, Optional[np.ndarray]]:
    ok, frame = stream.cap.read()
    if ok and frame is not None:
        return True, frame

    if stream.is_video and loop:
        stream.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok2, frame2 = stream.cap.read()
        if ok2 and frame2 is not None:
            return True, frame2

    return False, None


def _clamp_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _select_closest_car_rear(dets) -> Optional[object]:
    """Pick the closest car_rear using bbox bottom-y (largest is closest)."""
    cand = [d for d in dets if d.class_name == "car_rear" and d.bbox is not None]
    if not cand:
        return None

    def bottom_y(det):
        x1, y1, x2, y2 = det.bbox.xyxy_int()
        return y2

    return max(cand, key=bottom_y)


def _select_largest_traffic_light(dets) -> Optional[object]:
    cand = [d for d in dets if d.class_name == "traffic_light" and d.bbox is not None]
    if not cand:
        return None

    def area(det):
        return float(det.bbox.w * det.bbox.h)

    return max(cand, key=area)


def _traffic_light_state(frame_bgr: np.ndarray, tl_det) -> Tuple[str, float, float]:
    """Return (state, red_ratio, green_ratio). state in {'RED','GREEN','UNKNOWN'}.

    Uses simple HSV color ratio inside traffic_light bbox (and polygon mask if available).
    """
    if tl_det is None or tl_det.bbox is None:
        return "UNKNOWN", 0.0, 0.0

    h_img, w_img = frame_bgr.shape[:2]
    x1, y1, x2, y2 = tl_det.bbox.xyxy_int()
    x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, w_img, h_img)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN", 0.0, 0.0

    valid = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8) * 255
    if getattr(tl_det, "mask", None) is not None and getattr(tl_det.mask, "polygon", None):
        poly = np.array([[int(round(p.x)), int(round(p.y))] for p in tl_det.mask.polygon], dtype=np.int32)
        poly[:, 0] -= x1
        poly[:, 1] -= y1
        valid = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        cv2.fillPoly(valid, [poly.reshape((-1, 1, 2))], 255)

    valid_pixels = int(np.count_nonzero(valid))
    if valid_pixels < 50:
        return "UNKNOWN", 0.0, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    green = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))

    red = cv2.bitwise_and(red, valid)
    green = cv2.bitwise_and(green, valid)

    red_cnt = int(np.count_nonzero(red))
    green_cnt = int(np.count_nonzero(green))

    red_ratio = red_cnt / max(valid_pixels, 1)
    green_ratio = green_cnt / max(valid_pixels, 1)

    if red_cnt >= 30 and red_ratio >= 0.02 and red_ratio >= (green_ratio * 1.2 + 1e-6):
        return "RED", red_ratio, green_ratio
    if green_cnt >= 30 and green_ratio >= 0.02 and green_ratio >= (red_ratio * 1.2 + 1e-6):
        return "GREEN", red_ratio, green_ratio
    return "UNKNOWN", red_ratio, green_ratio


def _build_tl_debug_view(
    frame_bgr: np.ndarray,
    tl_det,
    tl_state_raw: str,
    tl_is_red: bool,
    red_ratio: float,
    green_ratio: float,
    obs_lane: Optional[str],
    obs_ahead_dist: Optional[float],
    mode: str,
    active_lane: str,
    avoid_trigger_px: int,
    tl_y: Optional[float] = None,
    car_det: Optional[object] = None,
) -> np.ndarray:
    vis = frame_bgr.copy()
    h_img, w_img = vis.shape[:2]

    tl_present = (tl_det is not None and getattr(tl_det, "bbox", None) is not None)
    if tl_present:
        x1, y1, x2, y2 = tl_det.bbox.xyxy_int()
        x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, w_img, h_img)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # [NEW] Car Rear Bounding Box (Cyan)
    car_y_str = "N/A"
    if car_det is not None and getattr(car_det, "bbox", None) is not None:
        xc1, yc1, xc2, yc2 = car_det.bbox.xyxy_int()
        xc1, yc1, xc2, yc2 = _clamp_xyxy(xc1, yc1, xc2, yc2, w_img, h_img)
        cv2.rectangle(vis, (xc1, yc1), (xc2, yc2), (255, 255, 0), 2)
        car_y_str = f"{yc2}"

    if (not tl_present) or (tl_state_raw not in ("RED", "GREEN")):
        disp_state = "NOT DETECTED"
    else:
        disp_state = tl_state_raw

    badge_w, badge_h = 220, 60
    pad = 10

    if disp_state == "RED":
        badge_color = (0, 0, 255)
        text_color = (255, 255, 255)
    elif disp_state == "GREEN":
        badge_color = (0, 200, 0)
        text_color = (0, 0, 0)
    else:
        badge_color = (80, 80, 80)
        text_color = (255, 255, 255)

    cv2.rectangle(vis, (pad, pad), (pad + badge_w, pad + badge_h), badge_color, -1)
    cv2.putText(vis, disp_state, (pad + 12, pad + 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

    stop_on = bool(tl_is_red)
    stop_color = (0, 0, 255) if stop_on else (0, 200, 0)
    cv2.rectangle(vis, (pad, pad + badge_h + 8), (pad + badge_w, pad + 2 * badge_h + 8), stop_color, -1)
    cv2.putText(
        vis,
        "STOP" if stop_on else "GO",
        (pad + 12, pad + badge_h + 8 + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )

    obs_present = (obs_lane is not None and obs_ahead_dist is not None)
    trig = False
    if obs_present:
        try:
            trig = (0.0 < float(obs_ahead_dist) < float(avoid_trigger_px))
        except Exception:
            trig = False

    y0 = pad + 2 * badge_h + 8 + 35
    line = 32
    cv2.putText(vis, f"TL_RAW={tl_state_raw} Y={tl_y if tl_y else 'N/A'} r={red_ratio:.3f} g={green_ratio:.3f}", (pad, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis, f"LANE={active_lane}  MODE={mode}  CarY={car_y_str}", (pad, y0 + line),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if obs_present:
        cv2.putText(
            vis,
            f"OBSTACLE: DETECTED  lane={obs_lane}  dist={float(obs_ahead_dist):.0f}px  trig={int(trig)}",
            (pad, y0 + 2 * line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    else:
        cv2.putText(vis, "OBSTACLE: NONE", (pad, y0 + 2 * line),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    right_x1 = max(pad, w_img - pad - badge_w)
    obs_badge_color = (0, 140, 255) if obs_present else (80, 80, 80)
    cv2.rectangle(vis, (right_x1, pad), (right_x1 + badge_w, pad + badge_h), obs_badge_color, -1)
    cv2.putText(vis, "OBSTACLE" if obs_present else "NO OBS", (right_x1 + 12, pad + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    changing = (mode == "CHANGING")
    ch_color = (255, 0, 255) if changing else (80, 80, 80)
    cv2.rectangle(vis, (right_x1, pad + badge_h + 8), (right_x1 + badge_w, pad + 2 * badge_h + 8), ch_color, -1)
    cv2.putText(vis, "CHANGING" if changing else "STABLE", (right_x1 + 12, pad + badge_h + 8 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return vis


def _warp_point(x: float, y: float, M: np.ndarray) -> Tuple[float, float]:
    pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def _build_blended_lane(
    edge_from: np.ndarray,
    edge_to: np.ndarray,
    y_samples: List[int],
    y_start: float,
    y_end: float,
) -> Optional[object]:
    """Create a blended LaneInfo by mixing lane centers from edge_from->edge_to."""
    from win_autodrive.messages import LaneInfo, TargetPoint

    if edge_from is None or edge_to is None:
        return None

    tps: List[TargetPoint] = []
    denom = float(max(1.0, (y_start - y_end)))

    for y in y_samples:
        xf = lane_center_x(edge_from, y=y, thickness=100)
        xt = lane_center_x(edge_to, y=y, thickness=100)
        if xf is None or xt is None:
            continue
        t = (y_start - float(y)) / denom
        t = float(np.clip(t, 0.0, 1.0))
        xb = (1.0 - t) * float(xf) + t * float(xt)
        tps.append(TargetPoint(x=int(round(xb)), y=int(y)))

    if not tps:
        return None

    return LaneInfo(slope=0.0, target_points=tps)


def _default_bird_cfg(w: int, h: int) -> BirdEyeConfig:
    src = np.array([[200, 350], [1080, 350], [1280, 720], [0, 720]], dtype=np.float32)
    dst = np.array([[520, 0], [780, 0], [780, 720], [520, 720]], dtype=np.float32)
    sx = w / 1280.0
    sy = h / 720.0
    src[:, 0] *= sx
    src[:, 1] *= sy
    dst[:, 0] *= sx
    dst[:, 1] *= sy
    return BirdEyeConfig(src=src, dst=dst, cut_row=int(300 * sy))


def main() -> None:
    import sys
    import torch
    print(f"[DEBUG] Python Executable: {sys.executable}")
    print(f"[DEBUG] Torch CUDA Available: {torch.cuda.is_available()}")

    ap = argparse.ArgumentParser(
        description="YOLOv8+차선추출+경로+조향 (2개 입력: lane_stream=lane1/lane2, obj_stream=traffic_light/car_rear)"
    )

    # ---- Inputs ----
    ap.add_argument("--cam_lane", type=int, default=1, help="차선용 입력 인덱스 (lane1/lane2). video_lane 없을 때 사용")
    ap.add_argument("--cam_obj", type=int, default=2, help="신호등/장애물용 입력 인덱스 (traffic_light/car_rear). video_obj 없을 때 사용")

    ap.add_argument("--cam", type=int, default=None, help="(호환용) --cam_lane과 동일")

    ap.add_argument("--video_lane", default=None, help="(옵션) 차선용 동영상 파일. 지정 시 cam_lane 대신 사용")
    ap.add_argument("--video_obj", default=None, help="(옵션) 객체/신호용 동영상 파일. 지정 시 cam_obj 대신 사용")
    ap.add_argument("--video", default=None, help="(호환용) --video_lane과 동일")

    # Capture settings (best-effort; video에는 대부분 무시됨)
    ap.add_argument("--width", type=int, default=1280, help="캡처 가로 해상도(권장 1280)")
    ap.add_argument("--height", type=int, default=720, help="캡처 세로 해상도(권장 720)")
    ap.add_argument("--fps", type=int, default=30, help="캡처 FPS(권장 30)")

    # ---- Models ----
    ap.add_argument("--model", default="best.pt", help="(호환용) 모델 경로. model_lane/model_obj 미지정 시 사용")
    ap.add_argument("--model_lane", default="2_cam1_lane_seg.pt", help="차선(lane1/lane2) 세그 모델(.pt). 미지정 시 --model 사용")
    ap.add_argument("--model_obj", default="2_cam2_obj_det.pt", help="객체(traffic_light/car_rear) 디텍 모델(.pt). 미지정 시 --model 사용")
    ap.add_argument("--device", default="cuda:0", help="YOLO 디바이스: cpu 또는 cuda:0")
    ap.add_argument("--conf", type=float, default=0.25, help="(공통) YOLO conf. conf_lane/conf_obj 미지정 시 사용")
    ap.add_argument("--conf_lane", type=float, default=0.25, help="(옵션) 차선 모델 conf")
    ap.add_argument("--conf_obj", type=float, default=0.35, help="(옵션) 객체 모델 conf")
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO 추론 해상도 (예: 640, 320)")

    # ---- Control / Serial ----
    ap.add_argument("--lane_class", default="lane2", help="시작 차선 선택: lane1 또는 lane2")
    ap.add_argument("--port", default=None, help="UART 포트 (예: COM7). 미지정 시 전송 안 함")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--send_hz", type=float, default=20.0)

    ap.add_argument("--loop", action="store_true", help="동영상 끝나면 처음으로 되감기")
    args = ap.parse_args()

    # ---- Resolve aliases ----
    cam_lane_idx = int(args.cam_lane)
    if args.cam is not None:
        cam_lane_idx = int(args.cam)

    video_lane = args.video_lane if args.video_lane is not None else None
    if args.video is not None:
        video_lane = args.video
    video_obj = args.video_obj if args.video_obj is not None else None

    model_lane_path = args.model_lane if args.model_lane else args.model
    model_obj_path = args.model_obj if args.model_obj else args.model
    conf_lane = args.conf_lane if args.conf_lane is not None else args.conf
    conf_obj = args.conf_obj if args.conf_obj is not None else args.conf

    # ---- Open streams (camera or video) ----
    stream_lane = _open_stream(cam_lane_idx, video_lane, args.width, args.height, args.fps)
    stream_obj = _open_stream(int(args.cam_obj), video_obj, args.width, args.height, args.fps)

    if not stream_lane.cap.isOpened():
        raise RuntimeError("Lane stream(VideoCapture) open 실패")
    if not stream_obj.cap.isOpened():
        raise RuntimeError("Object stream(VideoCapture) open 실패")

    # Read one frame to lock resolutions (and to guarantee paused mode has frames)
    ok_lane0, frame_lane = _read_stream(stream_lane, args.loop)
    ok_obj0, frame_obj = _read_stream(stream_obj, args.loop)
    if (not ok_lane0) or frame_lane is None:
        raise RuntimeError("차선 입력 첫 프레임 읽기 실패")
    if (not ok_obj0) or frame_obj is None:
        raise RuntimeError("객체/신호 입력 첫 프레임 읽기 실패")

    h, w = frame_lane.shape[:2]
    h_obj, w_obj = frame_obj.shape[:2]

    bird_cfg = _default_bird_cfg(w, h)
    M_bird = cv2.getPerspectiveTransform(bird_cfg.src.astype(np.float32), bird_cfg.dst.astype(np.float32))

    yolo_lane = YoloV8(YoloConfig(model_path=model_lane_path, device=args.device, conf=conf_lane,
                                 iou=args.iou, max_det=args.max_det, imgsz=args.imgsz))
    yolo_obj = YoloV8(YoloConfig(model_path=model_obj_path, device=args.device, conf=conf_obj,
                                iou=args.iou, max_det=args.max_det, imgsz=args.imgsz))

    print("\n[MODEL_LANE_PATH]", model_lane_path)
    print("[LANE_CLASS_NAMES]", yolo_lane.class_names())

    print("\n[MODEL_OBJ_PATH ]", model_obj_path)
    print("[OBJ_CLASS_NAMES ]", yolo_obj.class_names(), "\n")

    path_cfg = PathConfig(sample_count=40)
    motion_cfg = MotionConfig(max_steering=7, base_speed=255, steering_gain=1.0)

    # UART 포트 자동 선택 로직 (COM9 -> COM11)
    candidates = []
    if args.port is not None:
        candidates.append(args.port)
    else:
        # 우선순위: COM9 -> COM11
        candidates = ["COM9", "COM11"]

    ser = None
    connected_port = None

    for c_port in candidates:
        try:
            print(f"[UART] Connecting to {c_port}...")
            # SerialSender 생성자에서 실제 연결 시도 (실패 시 예외 발생)
            temp_ser = SerialSender(SerialConfig(port=c_port, baud=args.baud, timeout=0.1))
            
            # 예외가 없으면 연결 성공으로 간주
            ser = temp_ser
            connected_port = c_port
            print(f"[UART] Connected to {connected_port}")
            break
        except Exception as e:
            print(f"[UART] Connection failed for {c_port}: {e}")

    # 연결 실패 또는 포트 미지정 시: 더미(None) 모드로 동작
    if ser is None:
        print("[UART] No valid UART port found. Disabling serial transmission.")
        ser = SerialSender(SerialConfig(port=None))
        args.port = None
    else:
        # 연결 성공 시 args.port 업데이트 (이후 로직에서 사용됨)
        args.port = connected_port

    if args.port is not None:
        print("Arduino 연결 대기 (2초)...")
        time.sleep(2.0)

    viz = Visualizer(VizConfig(window_name="view"))

    cv2.namedWindow(TL_WIN_NAME, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(TL_WIN_NAME, 640, 360)
    except Exception:
        pass

    paused = False
    last_send_t = 0.0

    # ---- Mission state ----
    active_lane = args.lane_class if args.lane_class in LANE_CLASSES else "lane2"
    mode = "FOLLOW"  # FOLLOW or CHANGING
    change_from = "lane2"
    change_from = "lane2"
    change_to = "lane1"
    change_done_cnt = 0
    has_changed_lane = False  # 한번이라도 차선 변경(CHANGING) 모드에 진입했는지 여부

    # Traffic light debounce
    red_score = 3  # 0..3
    
    # [NEW] Stop -> Go Delay (빨간불에서 초록불로 바뀔 때만 1초 지연)
    tl_is_red = False # 실제 제어 상태
    last_red_time = 0.0
    GO_DELAY = 1.0

    # Tunables (pixel units in bird-eye image)
    NUM_Y = 20
    Y_MAX = 180
    target_rows = np.linspace(0, Y_MAX, NUM_Y)
    y_samples = [int(bird_cfg.cut_row + dy) for dy in target_rows]
    y_car = int(h - 10)
    x_car = float(w / 2.0)
    avoid_trigger_px = int(h * 0.40)
    done_err_px = int(w * 0.04)
    done_need_frames = 5

    # [NEW] Traffic Light Y-range Filter
    # 신호등 중심 y좌표가 이 범위 내에 있을 때만 빨간불 인식(멈춤) 수행
    TL_Y_MIN = 50       # 최소 y (화면 상단이 0)
    TL_Y_MAX = 328     # 최대 y (화면 하단이 h). 필요에 따라 조정 (예: 360)

    driving_active = False  # Start in stopped state

    try:
        while True:
            if not paused:
                @profile("CameraRead_Lane")
                def read_lane():
                    return _read_stream(stream_lane, args.loop)
                ok_lane, new_lane = read_lane()

                @profile("CameraRead_Obj")
                def read_obj():
                    return _read_stream(stream_obj, args.loop)
                ok_obj, new_obj = read_obj()

                if (not ok_lane) or new_lane is None:
                    break
                if (not ok_obj) or new_obj is None:
                    break

                frame_lane = new_lane
                frame_obj = new_obj

            # 1) YOLO detect
            @profile("YOLO_Lane")
            def run_yolo_lane():
                return yolo_lane.predict(frame_lane)
            dets_lane_all = run_yolo_lane()
            dets_lane = _filter_by_class(dets_lane_all, LANE_CLASSES)

            @profile("YOLO_Obj")
            def run_yolo_obj():
                return yolo_obj.predict(frame_obj)
            dets_obj_all = run_yolo_obj()
            dets_obj = _filter_by_class(dets_obj_all, OBJ_CLASSES)

            # 2) Traffic light state
            tl_det = _select_largest_traffic_light(dets_obj)
            tl_y = None
            
            if tl_det is not None and tl_det.bbox is not None:
                _, y1_tl, _, y2_tl = tl_det.bbox.xyxy_int()
                tl_y = (y1_tl + y2_tl) / 2.0
            
            # [Modified] 먼저 상태(RED/GREEN)를 판단한 뒤, 제어 로직에서만 Y범위 체크
            # 이렇게 해야 화면에는 RED라고 뜨는데 멈추지 않는 상황(범위 밖)을 모니터링 가능
            tl_state_raw, red_ratio, green_ratio = _traffic_light_state(frame_obj, tl_det)
            
            valid_y = False
            if tl_y is not None:
                if TL_Y_MIN <= tl_y <= TL_Y_MAX:
                    valid_y = True

            # RED이고 + 범위 내에 있을 때만 점수 증가
            if tl_state_raw == "RED" and valid_y:
                red_score = min(3, red_score + 1)
            else:
                # RED가 아니거나 or RED여도 범위 밖이면 점수 감소
                red_score = max(0, red_score - 1)
            
            # 1) 현재 프레임 기준 Red 여부 (Debounce 완료된 값)
            current_raw_red = (red_score >= 2)

            # 2) Logic: Red이면 즉시 반영. Green이면 1초 대기 후 해제.
            if current_raw_red:
                tl_is_red = True
                last_red_time = time.time()
            else:
                # 현재는 Green/Unknown 상태
                # 마지막 Red 시각으로부터 1초가 지났는지 확인
                if (time.time() - last_red_time) < GO_DELAY:
                    tl_is_red = True
                else:
                    tl_is_red = False

            # 3) Lane extraction
            edge_orig1, edge_bird1, lane1 = extract_lane_info(
                detections=dets_lane,
                frame_h=h,
                frame_w=w,
                bird_cfg=bird_cfg,
                lane_class_name="lane1",
                edge_thickness=2,
                target_rows=target_rows,
            )
            edge_orig2, edge_bird2, lane2 = extract_lane_info(
                detections=dets_lane,
                frame_h=h,
                frame_w=w,
                bird_cfg=bird_cfg,
                lane_class_name="lane2",
                edge_thickness=2,
                target_rows=target_rows,
            )

            edge_orig = np.maximum(edge_orig1, edge_orig2)
            edge_bird_roi = np.maximum(edge_bird1, edge_bird2)

            lanes = {"lane1": lane1, "lane2": lane2}
            edges_bird = {"lane1": edge_bird1, "lane2": edge_bird2}

            if active_lane not in LANE_CLASSES:
                active_lane = "lane2"

            # 4) Obstacle (car_rear) heuristic mapping (cam_obj -> cam_lane)
            car_det = _select_closest_car_rear(dets_obj)
            obs_lane = None
            obs_ahead_dist = None
            if car_det is not None and car_det.bbox is not None:
                x1o, y1o, x2o, y2o = car_det.bbox.xyxy_int()
                x_obj = float((x1o + x2o) * 0.5)
                y_obj = float(y2o)

                x_lane = x_obj * (float(w) / max(1.0, float(w_obj)))
                y_lane = y_obj * (float(h) / max(1.0, float(h_obj)))
                x_obs_bev, y_obs_bev = _warp_point(x_lane, y_lane, M_bird)

                # [Modified] BEV 대신 단순히 화면상의 Y 거리로 판단
                # distance = (화면 하단) - (객체 하단). 화면 밖으로 벗어나면(가까우면) 0으로 처리
                obs_ahead_dist = max(0.0, float(h) - y_lane)

                # [Modified] y_car보다 더 가깝게(아래에) 있어도 감지되도록 조건 완화
                if y_obs_bev >= float(bird_cfg.cut_row):
                     # lookup Y는 이미지 범위를 넘지 않도록 클램핑
                     look_y = int(min(float(h) - 1, round(y_obs_bev)))
                     
                     x1c = lane_center_x(edges_bird["lane1"], y=look_y, thickness=120) if edges_bird["lane1"] is not None else None
                     x2c = lane_center_x(edges_bird["lane2"], y=look_y, thickness=120) if edges_bird["lane2"] is not None else None
                     
                     if x1c is not None and x2c is not None:
                         obs_lane = "lane1" if abs(x_obs_bev - float(x1c)) < abs(x_obs_bev - float(x2c)) else "lane2"

            # 5) Decide lane change (only when not red)
            if (not tl_is_red) and obs_lane is not None and obs_ahead_dist is not None:
                if obs_ahead_dist <= 20:
                    if mode == "FOLLOW" and obs_lane == active_lane:
                        target = "lane1" if active_lane == "lane2" else "lane2"
                        if lanes.get(target) is not None:
                            mode = "CHANGING"
                            has_changed_lane = True
                            change_from = active_lane
                            change_to = target
                            change_done_cnt = 0
                            print(f"[LaneChange Trigger] Obs in {obs_lane}, switching to {target}")

            # [Modified] 현재 차선 놓쳤을 때 반대 차선으로 갈아타는 로직 (Fallback)
            # --> 장애물 회피가 먼저 발동되면(CHANGING) 굳이 Fallback을 안 타도 됨.
            if mode == "FOLLOW":
                 # 빨간불 정지 중일 때는(차선 인식이 불안정할 수 있음) 굳이 바꾸지 않도록 보호
                if (not tl_is_red) and (lanes.get(active_lane) is None):
                     other = "lane1" if active_lane == "lane2" else "lane2"
                     if lanes.get(other) is not None:
                         # [User Request] 첫 차선 변경 전에는 무조건 lane2 유지
                         # has_changed_lane이 False이면(아직 회피 기동 안함)
                         # 그리고 현재 lane2인데 놓친거면 -> lane1으로 Fallback 금지
                         if (not has_changed_lane) and (active_lane == "lane2"):
                             pass 
                         else:
                             active_lane = other
                             print(f"[Fallback] Lost {active_lane}, switching active to {other}")

            # 6) Build lane to follow
            lane_for_path = None
            if mode == "FOLLOW":
                lane_for_path = lanes.get(active_lane)
            else:
                y_start = float(y_car)
                y_end = float(bird_cfg.cut_row)
                blended = _build_blended_lane(
                    edge_from=edges_bird.get(change_from),
                    edge_to=edges_bird.get(change_to),
                    y_samples=y_samples,
                    y_start=y_start,
                    y_end=y_end,
                )
                lane_for_path = blended if blended is not None else lanes.get(change_to)

                x_t = lane_center_x(edges_bird.get(change_to), y=y_car, thickness=140) if edges_bird.get(change_to) is not None else None
                if x_t is not None:
                    if abs(float(x_t) - x_car) < float(done_err_px):
                        change_done_cnt += 1
                    else:
                        change_done_cnt = 0
                    if change_done_cnt >= int(done_need_frames):
                        mode = "FOLLOW"
                        active_lane = change_to
                        change_done_cnt = 0

            # 7) Path + motion
            path = None
            cmd = None

            # [Modified] 항상 경로 생성 시도 (신호등 상관없이 시각화/조향 유지)
            if lane_for_path is not None:
                car_center = (x_car, float(y_car))
                path = plan_path(lane_for_path, path_cfg, car_center=car_center)
                cmd = plan_motion(path, motion_cfg)
            else:
                cmd = MotionCommand(steering=motion_cfg.center_output, left_speed=0, right_speed=0)

            # [Modified] 빨간불이면 속도만 0으로 강제 (경로는 유지)
            if tl_is_red:
                if cmd is None:  # 혹시나 예외 상황
                     cmd = MotionCommand(steering=motion_cfg.center_output, left_speed=0, right_speed=0)
                cmd.left_speed = 0
                cmd.right_speed = 0

            # [NEW] Driving active control
            if not driving_active:
                # Force stop and center steering when not active
                # motion_cfg might be defined inside checking scope, but it is defined above in main scope (line 419)
                center_val = motion_cfg.center_output
                cmd = MotionCommand(steering=int(center_val), left_speed=0, right_speed=0)

            # Serial send/read (optional)
            now = time.time()
            if ser is not None and cmd is not None:
                if (now - last_send_t) >= (1.0 / max(args.send_hz, 1e-6)):
                    ser.send(cmd)
                    last_send_t = now

                line = ser.read()
                if line:
                    pass

            # TL debug window
            try:
                tl_view = _build_tl_debug_view(
                    frame_obj,
                    tl_det,
                    tl_state_raw,
                    tl_is_red,
                    red_ratio,
                    green_ratio,
                    obs_lane,
                    obs_ahead_dist,
                    mode,
                    active_lane,
                    avoid_trigger_px,
                    tl_y=tl_y,
                    car_det=car_det,
                )
                cv2.imshow(TL_WIN_NAME, tl_view)
            except Exception:
                pass

            # Overlay mission status on lane stream
            frame_vis = frame_lane.copy()
            tl_y_str = f"{tl_y:.0f}" if tl_y is not None else "nan"
            cv2.putText(frame_vis, f"TL={tl_state_raw} Y={tl_y_str} red={int(tl_is_red)} r={red_ratio:.3f} g={green_ratio:.3f}", (15, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            dist_disp = (obs_ahead_dist if obs_ahead_dist is not None else -1.0)
            cv2.putText(frame_vis, f"Mode={mode} lane={active_lane} obs_lane={obs_lane} dist={dist_disp:.0f}", (15, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            status_str = "Press SPACE to Stop" if driving_active else "Press SPACE to Start"
            status_color = (0, 255, 0) if driving_active else (0, 0, 255)
            cv2.putText(frame_vis, f"{status_str}", (15, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            @profile("Visualizer")
            def run_viz():
                return viz.show(frame_vis, edge_orig, edge_bird_roi, bird_cfg, lane_for_path, path, cmd)

            key = run_viz()
            if key == ord("q"):
                break
            if key == ord("p"):  # Changed from space to p for pause
                paused = not paused
            if key == ord(" "):  # Space for start/stop
                driving_active = not driving_active
                print(f"[System] Driving Active: {driving_active}")

    finally:
        print_stats()

        # 종료 시 모터 정지 (시리얼 사용 시)
        if ser is not None:
            try:
                center = motion_cfg.center_output if "motion_cfg" in locals() else 60
                stop_cmd = MotionCommand(steering=int(center), left_speed=0, right_speed=0)
                ser.send(stop_cmd)
                time.sleep(0.1)
            except Exception:
                pass

        try:
            stream_lane.cap.release()
        except Exception:
            pass
        try:
            stream_obj.cap.release()
        except Exception:
            pass
        try:
            if ser is not None:
                ser.close()
        except Exception:
            pass
        try:
            viz.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
