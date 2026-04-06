from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .messages import MotionCommand, PathPlanningResult
except ImportError:
    # Run as script
    from messages import MotionCommand, PathPlanningResult


@dataclass
class MotionConfig:
    """MotionConfig: 조향/속도 명령 파라미터.
    
    steering(조향): Servo 값 (41 ~ 79), 중앙값 60.
    Pure Pursuit 알고리즘을 위한 파라미터 포함.
    """

    max_steering: int = 7 # 사용 안 함 (레거시)
    
    # Servo Output Constraints
    min_output: int = 41
    max_output: int = 79
    center_output: int = 60
    
    # Pure Pursuit Parameters
    lookahead_index: int = 13 # 경로 배열에서 몇 번째 앞의 점을 볼 것인가 (0이 가장 먼 점일 수 있으니 주의 필요)
                              # path.py의 plan_path는 sorted(pts, key=lambda p: p[1])로 정렬함.
                              # y가 작은 순서대로 정렬됨 (이미지 상단이 y=0이라면 먼 곳이 y=0 근처)
                              # 따라서 index 0이 가장 먼 점, index -1이 차와 가장 가까운 점(보닛 위치)
    
    base_speed: int = 50
    steering_gain: float = 0.9  # 각도(degree) 당 서보 변화량 gain 튜닝 필요


def plan_motion(path: PathPlanningResult, cfg: MotionConfig) -> MotionCommand:
    """Convert planned path into a servo steering command using Pure Pursuit."""

    if len(path.x_points) < 2 or len(path.y_points) < 2:
        # 경로가 없으면 직진(중앙값) 혹은 정지
        return MotionCommand(steering=cfg.center_output, left_speed=0, right_speed=0)

    # Path points are sorted by Y ascending.
    # Y=0 is top of image (far), Y=720 is bottom (close).
    # path.x_points[0], path.y_points[0] -> Furthest point (Top)
    # path.x_points[-1], path.y_points[-1] -> Closest point (Car bonnet)
    
    # Car position (image coordinates)
    car_x = path.x_points[-1]
    car_y = path.y_points[-1]

    # Lookahead point selection
    # We want a point somewhat ahead of the car.
    # Since index -1 is the car, we go backwards or pick a fixed index from start?
    # sorted by y ascending means small y (far) is at index 0.大 y (close) is at index -1.
    # Let's pick a point 'lookahead_index' away from the car (towards index 0)
    
    idx_target = max(0, len(path.x_points) - 1 - cfg.lookahead_index)
    
    target_x = path.x_points[idx_target]
    target_y = path.y_points[idx_target]

    # Calculate angle to target
    # Image frame: x right, y down.
    # Car heading is assumed usually "Up" matches decreasing Y.
    # Vector Car->Target: (dx, dy) = (target_x - car_x, target_y - car_y)
    # Usually dy is negative (target is higher/further up).
    
    dx = target_x - car_x
    dy = target_y - car_y # Negative value usually

    # Angle relative to vertical (up) axis.
    # If dx > 0 (Right), angle should be positive?
    # atan2(x, -y):
    #   Forward (dx=0, dy=-10) -> atan2(0, 10) = 0
    #   Right (dx=10, dy=-10) -> atan2(10, 10) = 45 deg
    #   Left (dx=-10, dy=-10) -> atan2(-10, 10) = -45 deg
    
    angle_rad = math.atan2(dx, -dy) # -dy makes "up" vector positive for atan2 reference
    angle_deg = math.degrees(angle_rad)

    # Steering Command Calculation
    # Assumption: 
    #   High Servo Value (> 60) -> Left Turn
    #   Low Servo Value (< 60) -> Right Turn
    #   (Based on user's prompt: steer 41~79, center 60)
    #   
    # If angle_deg > 0 (Target is Right): We need to turn Right using Low Value.
    #   => servo = 60 - (Positive Angle * gain)
    # If angle_deg < 0 (Target is Left): We need to turn Left using High Value.
    #   => servo = 60 - (Negative Angle * gain) = 60 + (Positive Amount)
    
    # Error term calculation
    error = angle_deg
    
    raw_steer = cfg.center_output - (error * cfg.steering_gain)
    
    # Clamp
    steer_val = int(np.clip(raw_steer, cfg.min_output, cfg.max_output))
    
    # Speed Control
    # User requested max speed fixed (no differential)
    sp = int(cfg.base_speed)
    
    return MotionCommand(steering=steer_val, left_speed=sp, right_speed=sp)


def test_straight_line():
    print("Testing Straight Line...")
    x_pts = [640.0] * 10
    y_pts = [float(i * 72) for i in range(10)]
    path = PathPlanningResult(x_points=x_pts, y_points=y_pts)
    cfg = MotionConfig()
    cmd = plan_motion(path, cfg)
    print(f"Straight Line Result: {cmd}")
    assert abs(cmd.steering - 60) < 5

def test_left_turn():
    print("\nTesting Left Turn...")
    x_pts = [float(i * 71) for i in range(10)]
    y_pts = [float(i * 72) for i in range(10)]
    path = PathPlanningResult(x_points=x_pts, y_points=y_pts)
    cfg = MotionConfig()
    cmd = plan_motion(path, cfg)
    print(f"Left Turn Result: {cmd}")
    assert cmd.steering > 60

def test_right_turn():
    print("\nTesting Right Turn...")
    x_pts = [1280.0, 1200.0, 1100.0, 1000.0, 900.0, 800.0, 700.0, 640.0, 640.0, 640.0]
    y_pts = [float(i * 72) for i in range(10)]
    path = PathPlanningResult(x_points=x_pts, y_points=y_pts)
    cfg = MotionConfig()
    cmd = plan_motion(path, cfg)
    print(f"Right Turn Result: {cmd}")
    assert cmd.steering < 60

if __name__ == "__main__":
    test_straight_line()
    test_left_turn()
    test_right_turn()
    print("\nAll tests passed!")
