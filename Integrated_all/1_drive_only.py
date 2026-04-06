from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from win_autodrive.lane import BirdEyeConfig, extract_lane_info
from win_autodrive.messages import MotionCommand
from win_autodrive.motion import MotionConfig, plan_motion
from win_autodrive.path import PathConfig, plan_path
from win_autodrive.serial_io import SerialConfig, SerialSender
from win_autodrive.viz import VizConfig, Visualizer
from win_autodrive.yolo import YoloConfig, YoloV8
from win_autodrive.profiler import profile, print_stats


def _default_bird_cfg(w: int, h: int) -> BirdEyeConfig:
    # ROS2 lane_info_extractor_node.py의 src/dst를 동일하게 반영
    # src = np.array([[200, 350], [1000, 350], [1280, 720], [0, 720]], dtype=np.float32)
    src = np.array([[200, 350], [1080, 350], [1280, 720], [0, 720]], dtype=np.float32)
    dst = np.array([[520, 0], [780, 0], [780, 720], [520, 720]], dtype=np.float32)
    # 영상 해상도에 맞게 스케일링
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
    
    ap = argparse.ArgumentParser(description="ROS2 코드 기반 YOLOv8+차선추출+경로+조향을 Windows Python으로 실행")
    ap.add_argument("--video", default=None, help="비디오 파일 경로. 없으면 --cam 사용") # 기본: None (웹캠 사용)
    ap.add_argument("--cam", type=int, default=1, help="웹캠 인덱스 (기본 1: 외장 웹캠)")
    ap.add_argument("--model", default="1_drive_only.pt", help="YOLOv8 모델(.pt) 경로")
    ap.add_argument("--device", default="cuda:0", help="YOLO 디바이스: cpu 또는 cuda:0")
    ap.add_argument("--lane_class", default="lane2", help="차선(세그멘테이션) 클래스명")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max_det", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO 추론 해상도 (예: 640, 320)")

    ap.add_argument("--port", default=None, help="UART 포트 (예: COM7). 없으면 전송 안 함")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--send_hz", type=float, default=20.0)

    ap.add_argument("--loop", action="store_true", help="영상 끝나면 처음으로")
    ap.add_argument("--rotate_right", action="store_true", help="카메라 오른쪽으로 90도 회전")
    args = ap.parse_args()

    cap: cv2.VideoCapture
    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
    else:
        # Windows: cv2.CAP_DSHOW를 쓰면 카메라 초기화 속도가 획기적으로 빨라짐 (10s -> 0.5s)
        # 하지만 해상도 설정 등이 필요할 수 있으므로, 열고 나서 설정
        cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
        
        # DSHOW 사용 시 해상도가 기본값(640x480)으로 잡힐 수 있음.
        # 필요하다면 아래처럼 원하는 해상도를 직접 지정해야 함 (여기선 자동 리사이즈가 있어서 괜찮음)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("VideoCapture open 실패")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        # 일부 webcam에서 0으로 나오는 경우가 있어, 첫 프레임으로 추정
        ok, frm = cap.read()
        if not ok or frm is None:
            raise RuntimeError("첫 프레임 읽기 실패")
        h, w = frm.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if args.rotate_right:
        w, h = h, w

    bird_cfg = _default_bird_cfg(w, h)

    yolo = YoloV8(YoloConfig(model_path=args.model, device=args.device, conf=args.conf, iou=args.iou, max_det=args.max_det, imgsz=args.imgsz))
    path_cfg = PathConfig(sample_count=40)
    motion_cfg = MotionConfig(lookahead_index=13, base_speed=255, steering_gain=1.0)
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

    paused = False
    driving_active = False
    last_send_t = 0.0

    try:
        while True:
            if not paused:
                @profile("CameraRead")
                def read_cam():
                    return cap.read()
                ok, frame = read_cam()

                if not ok or frame is None:
                    if args.loop and args.video is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                    
                if args.rotate_right:
                    # 90도 시계방향 회전
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # 1) YOLO detect
            if not driving_active:
                # 정지 상태일 때: 조향 60(중앙), 속도 0
                # 주의: 사용자 요청에 따라 steering=60으로 설정 (기존 코드는 0이었으나 요청 반영)
                cmd = MotionCommand(steering=60, left_speed=0, right_speed=0)
            
                # 화면에 상태 표시
                cv2.putText(frame, "Press SPACE to Start", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Press SPACE to Stop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.2, (0, 255, 0), 3)

            detections = yolo.predict(frame)

            # 2) Lane extraction
            edge_orig, edge_bird_roi, lane = extract_lane_info(
                detections=detections,
                frame_h=h,
                frame_w=w,
                bird_cfg=bird_cfg,
                lane_class_name=args.lane_class,
                edge_thickness=2,
                target_rows=(0, 100, 150, 200),
            )

            # 3) Path plan
            path = None
            cmd = None
            if lane is not None:
                car_center = (w / 2.0, float(h - 10))
                path = plan_path(lane, path_cfg, car_center=car_center)
                cmd = plan_motion(path, motion_cfg)

            # [Rule] driving_active가 False이면 위에서 계산된 cmd를 무시하고 정지 명령을 덮어씀
            if not driving_active:
                 cmd = MotionCommand(steering=60, left_speed=0, right_speed=0)

            # 4) Serial send (rate-limited)
            now = time.time()
            if cmd is not None and args.port is not None:
                if (now - last_send_t) >= (1.0 / max(args.send_hz, 1e-6)):
                    ser.send(cmd)
                    last_send_t = now

            # 4-2) Serial read (Arduino feedback)
            if args.port is not None:
                line = ser.read()
                if line:
                    # print(f"[Arduino] {line}") # 속도 저하 방지
                    pass

            # 5) Visualization
            @profile("Visualizer")
            def run_viz():
                return viz.show(frame, edge_orig, edge_bird_roi, bird_cfg, lane, path, cmd)
                # return viz.show_simple(frame, lane, path, cmd)
            
            key = run_viz()
            if key == ord('q'):
                break
            if key == ord(' '):
                driving_active = not driving_active
            elif key == ord('p'):
                paused = not paused

    finally:
        print_stats()  # 프로그램 종료 시 성능 통계 출력
        
        # 안전 장치: 종료 시 모터 정지 명령 전송
        if ser is not None:
             try:
                 stop_cmd = MotionCommand(steering=0, left_speed=0, right_speed=0)
                 ser.send(stop_cmd)
                 time.sleep(0.1) # 전송 대기
             except Exception:
                 pass

        cap.release()
        ser.close()
        viz.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
