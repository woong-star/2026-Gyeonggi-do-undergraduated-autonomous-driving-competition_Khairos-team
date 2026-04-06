import cv2
cv = cv2
import math
import time
from enum import Enum, auto
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# [NEW] Import YOLO
try:
    from ultralytics import YOLO
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
except ImportError:
    print("[Warn] ultralytics not found. YOLO disabled.")
    YOLO = None


# Import win_autodrive modules for standard protocol
try:
    from win_autodrive.messages import MotionCommand
    from win_autodrive.serial_io import SerialConfig, SerialSender
    from Lib_LiDAR import libLIDAR as libLidar
except ImportError:
    try:
         sys.path.append("c:/ggdrive/parking") # Force path just in case /windrive 있는 폴더 상위
         from win_autodrive.messages import MotionCommand
         from win_autodrive.serial_io import SerialConfig, SerialSender
         libLidar = None
    except ImportError:

         print("Error: win_autodrive modules not found.")
         sys.exit(1)


# ==============================================================================
# Configuration
# ==============================================================================
LIDAR_PORT = 'COM3'
ARDUINO_PORT = 'COM9'
LIDAR_X_OFFSET = 0.4  # Lidar offset (User requested + direction)
# [NEW] Camera Config
CAMERA_INDEX = 1      # Default Camera Index (0=Webcam)

# [NEW] YOLO / BEV Config
MODEL_PATH = "3_parking.pt"
LANE_WIDTH_M = 2.0
LOOKAHEAD_M_YOLO = 1.0  # YOLO Pure Pursuit Lookahead
BEV_W, BEV_H = 500, 700
M_PER_PX_X = 0.01
M_PER_PX_Y = 0.01
CENTER_OFFSET_M = +LANE_WIDTH_M / 2.0
LANE_CLASS_NAME = "line"
WHEELBASE_M = 0.55  # Used for Pure Pursuit
MAX_STEER_DEG = 25.0

# BEV Homography Points (From parking_line.py / User Calibration)
SRC_PTS = np.float32([
    [  0, 142],  # left-top     (x=0, y avg)
    [638, 142],  # right-top    (x=638, y avg)
    [638, 477],  # right-bottom (x=638, y avg)
    [  0, 477],  # left-bottom  (x=0, y avg)
])

DST_PTS = np.float32([
    [ 50,   0],  # Left-Top     (중앙 250에서 -200, 4m 좌측)
    [450,   0],  # Right-Top    (중앙 250에서 +200, 4m 우측)
    [450, 700],  # Right-Bottom (차량 바로 앞, 우측)
    [ 50, 700],  # Left-Bottom  (차량 바로 앞, 좌측)
])

# ==============================================================================
# Helper Math / Logic Functions (From 3_lidar_roi_cluster_viz.py)
# ==============================================================================

def scan_to_points(angles_rad: np.ndarray, ranges_m: np.ndarray, 
                   min_dist: float, max_dist: float, 
                   min_angle: float, max_angle: float) -> np.ndarray:
    degrees = np.degrees(angles_rad)
    degrees = (degrees + 180) % 360 - 180
    
    mask = (
        np.isfinite(ranges_m) & 
        (ranges_m > min_dist) & (ranges_m < max_dist) &
        (degrees >= min_angle) & (degrees <= max_angle)
    )

    a = angles_rad[mask]
    r = ranges_m[mask]
    
    # Standard: x=r*cos(a), y=r*sin(a)
    x = r * np.cos(a)
    y = r * np.sin(a)
    
    return np.stack([x, y], axis=1)

def dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.array([], dtype=int)
    
    # 1. Vectorized Distance Matrix Calculation (N x N)
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    
    # 2. Find Neighbors matrix
    eps2 = eps * eps
    adj = d2 <= eps2  # Boolean adjacency matrix
    
    # 3. Identify Core Points
    n_neighbors = np.sum(adj, axis=1)
    core_mask = n_neighbors >= min_samples
    
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)
    
    # 4. Cluster Expansion (BFS/DFS)
    stack = []
    
    for i in range(n):
        if visited[i] or not core_mask[i]:
            continue
            
        labels[i] = cluster_id
        visited[i] = True
        stack.append(i)
        
        while stack:
            curr = stack.pop()
            
            # Get neighbors using boolean indexing
            nbrs_indices = np.where(adj[curr])[0]
            unvisited_nbrs = nbrs_indices[~visited[nbrs_indices]]
            
            if len(unvisited_nbrs) > 0:
                visited[unvisited_nbrs] = True
                labels[unvisited_nbrs] = cluster_id
                
                new_cores = unvisited_nbrs[core_mask[unvisited_nbrs]]
                stack.extend(new_cores.tolist())
                
        cluster_id += 1
        
    return labels

def find_closest_pair_between_clusters(pts1: np.ndarray, pts2: np.ndarray):
    if len(pts1) == 0 or len(pts2) == 0:
        return None, None, float('inf')

    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2) 
    min_idx = np.argmin(dist2)
    i, j = np.unravel_index(min_idx, dist2.shape)
    
    p1 = pts1[i]
    p2 = pts2[j]
    d = np.sqrt(dist2[i, j])
    return p1, p2, d

def get_center_line_from_corners(p1, p2):
    mid = (p1 + p2) / 2.0
    vec = p2 - p1
    dx, dy = vec
    
    nx, ny = -dy, dx
    norm = math.hypot(nx, ny)
    if norm < 1e-6:
        return None
    nx /= norm
    ny /= norm
    
    # Match reference: Ensure X is positive (Forward/Into spot?)
    # 3_lidar_roi_cluster_viz.py checks: if nx < 0: nx, ny = -nx, -ny
    if nx < 0:
        nx, ny = -nx, -ny
        
    a = -ny
    b = nx
    c = -(a * mid[0] + b * mid[1])
    return (a, b, c), mid

def calculate_sliding_target(line, vehicle_pos, lookahead):
    """
    Project vehicle_pos onto the line, then move forward by lookahead.
    line: (a, b, c) where ax + by + c = 0
    vehicle_pos: (x, y)
    lookahead: distance in meters
    """
    if line is None: return None
    a, b, c = line 
    
    # 1. Line Direction Vector (vx, vy)
    # Normal is (a, b). Direction is (b, -a)
    vx, vy = b, -a
    
    # Ensure Positive X (Into Parking Spot)
    if vx < 0: 
        vx, vy = -vx, -vy
        
    # 2. Project Vehicle Pos onto Line
    # distance from point to line d = (ax0 + by0 + c) / sqrt(a^2+b^2)
    # Since (a,b) is normalized in get_center_line, denom is 1.
    x0, y0 = vehicle_pos
    dist = a*x0 + b*y0 + c
    
    # Projected Point: P_proj = P - dist * Normal
    proj_x = x0 - a * dist
    proj_y = y0 - b * dist
    
    # 3. Apply Lookahead along the line
    target_x = proj_x + vx * lookahead
    target_y = proj_y + vy * lookahead
    
    # [Check] If target is BEHIND vehicle (x < 0), clamp to at least projection? 
    # Or strict sliding? Strict sliding ensures convergence.
    return np.array([target_x, target_y])

# def calculate_pure_pursuit_target(line, start_point, lookahead):
#    ... (Deprecating old one, using new one above)


def generate_arc_path(p_start, p_end):
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    Ld2 = dx*dx + dy*dy
    Ld = math.sqrt(Ld2)
    if Ld < 0.01:
        return np.array([p_start])
        
    alpha = math.atan2(dy, dx)
    # kappa = 2 * sin(alpha) / Ld
    kappa = 2.0 * math.sin(alpha) / Ld
    
    pts = []
    x, y, yaw = p_start[0], p_start[1], 0.0
    pts.append([x, y])
    ds = 0.05
    steps = int(Ld * 1.5 / ds)
    
    for _ in range(steps):
        x += ds * math.cos(yaw)
        y += ds * math.sin(yaw)
        yaw += ds * kappa
        pts.append([x, y])
        
    return np.array(pts)

def draw_line(ax, line, xlim, color='g', linestyle='--', linewidth=1):
    a, b, c = line
    # ax + by + c = 0 => y = (-ax - c) / b
    x_vals = np.array(xlim)
    if abs(b) > 1e-6:
        y_vals = (-a * x_vals - c) / b
        ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=linewidth)
    else:
        # Vertical line x = -c/a
        x_const = -c / a
        ax.axvline(x_const, color=color, linestyle=linestyle, linewidth=linewidth)

# ==============================================================================
# [NEW] YOLO / BEV Helper Functions
# ==============================================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_text(img, text, org, scale=0.7, thickness=2, color=(255,255,255)):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness+2, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness, cv.LINE_AA)

def get_homography():
    H = cv.getPerspectiveTransform(SRC_PTS, DST_PTS)
    H_inv = cv.getPerspectiveTransform(DST_PTS, SRC_PTS)
    return H, H_inv

def bev_warp(img, H):
    return cv.warpPerspective(img, H, (BEV_W, BEV_H), flags=cv.INTER_LINEAR)

def mask_from_yolo_seg(result, frame_shape, class_name=None):
    h, w = frame_shape[:2]
    if result.masks is None:
        return None

    masks = result.masks.data.cpu().numpy()  # (N, mh, mw) 0~1
    boxes = result.boxes
    if boxes is None or len(masks) == 0:
        return None

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    names = result.names  # dict: id->name

    candidates = []
    for i in range(len(masks)):
        name = names.get(int(cls_ids[i]), str(cls_ids[i]))
        if (class_name is None) or (name == class_name):
            m = masks[i]
            # resize to frame size
            m_resized = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
            m_bin = (m_resized > 0.5).astype(np.uint8) * 255
            area = int(m_bin.sum() / 255)
            candidates.append((area, m_bin, name))

    if len(candidates) == 0:
        return None

    # pick biggest
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def fit_lane_x_of_y(mask_bev):
    ys, xs = np.where(mask_bev > 0)
    if len(xs) < 200:
        return None

    valid = (ys > 50) & (ys < BEV_H-10)
    ys = ys[valid]
    xs = xs[valid]
    if len(xs) < 200:
        return None

    coeffs = np.polyfit(ys.astype(np.float32), xs.astype(np.float32), 2)
    return coeffs  # [a,b,c] for x = a*y^2 + b*y + c

def sample_center_path(coeffs, offset_m, y_start=80, y_end=None, step=10):
    if y_end is None:
        y_end = BEV_H - 20

    a, b, c = coeffs
    pts = []
    offset_px = offset_m / M_PER_PX_X  # meters -> pixels
    for y in range(int(y_start), int(y_end), int(step)):
        x_lane = a*y*y + b*y + c
        x_center = x_lane + offset_px
        pts.append((float(x_center), float(y)))
    return pts  # list of (x_px, y_px)

def pure_pursuit_steer(center_path_bev, lookahead_m):
    car_x = BEV_W / 2.0
    car_y = BEV_H - 1.0

    Ld_px = lookahead_m / M_PER_PX_Y
    target = None
    for (x, y) in center_path_bev:
        dx = x - car_x
        dy = car_y - y  # 전방(+)
        dist = math.sqrt(dx*dx + dy*dy)
        if dist >= Ld_px:
            target = (x, y, dx, dy, dist)
            break

    if target is None:
        return 0.0, None

    x, y, dx, dy, dist = target
    alpha = math.atan2(dx, dy)
    # pure pursuit
    delta = math.atan2(2.0 * WHEELBASE_M * math.sin(alpha), lookahead_m)
    delta_deg = math.degrees(delta)
    delta_deg = clamp(delta_deg, -MAX_STEER_DEG, MAX_STEER_DEG)
    return delta_deg, (int(x), int(y))

def overlay_path_on_bev(bev_img, path_pts, color=(255, 0, 255)):
    for i in range(1, len(path_pts)):
        p1 = (int(path_pts[i-1][0]), int(path_pts[i-1][1]))
        p2 = (int(path_pts[i][0]), int(path_pts[i][1]))
        cv.line(bev_img, p1, p2, color, 2)


class ParkingState(Enum):
    SCAN_GAP = auto()         
    REVERSE = auto()
    WAIT_HOLD = auto()
    REVERSE_CHECK_STOP = auto() # [NEW] Intermediate state for stopping check
    FINISHED = auto()
    FAIL = auto()
    
    # Stage 3 States
    S3_FWD_CHECK = auto()   # Check sensors while moving forward
    S3_FWD_WAIT = auto()    # 1s extra forward
    S3_RIGHT_TURN = auto()  # Hard right turn
    S3_STRAIGHT = auto()    # Straight to finish

    # [NEW] Stage 1 States
    S1_STRAIGHT = auto()    # 1. Forward until US detect (Pure Pursuit)
    S1_DECIDE = auto()      # 2. Stop & Decide Steer
    S1_CURVE = auto()       # 3. Move & Check Lidar (Custom ROI)
    S1_WAIT = auto()        # 4. Wait before Stage 2

    # [NEW] Input Mode (YOLO)
    INPUT_MODE = auto()

class VerticalParkingLidar:
    def __init__(self, max_range=2.0, angle_min=-90, angle_max=90):
        # Default start state
        self.state = ParkingState.S1_STRAIGHT
        self.start_time = time.time() # [NEW] Timer for start delay
        
        self.target_point: Optional[Tuple[float, float]] = None 
        
        # Lane Angle Data
        self.lane_angle: Optional[float] = None
        self.lane_angle_threshold = 57.0 
        
        # Timers
        self.state_timer = 0.0
        self.hold_start_time = 0.0
        
        # Ultrasonic Flags
        # self.flag_back_detect = False
        # self.flag_front_detect = False
        
        # Visualization Data (Lidar)
        self.viz_points = None
        self.viz_labels = None
        self.viz_corners = []
        self.viz_center_line = None
        self.viz_target = None
        self.viz_arc_path = None
        self.viz_s1_min_dist = None 
        self.vis_frame = None # [NEW] Shared Frame Buffer
        self.bev_frame = None # [NEW] Shared BEV Buffer
        
        # Config
        self.LIDAR_MAX_DIST = max_range 
        self.LIDAR_MAX_DIST = max_range 
        self.NORMAL_MAX_DIST = 4.0 # [UPDATED] Fixed to 4.0m for Reverse Mode 
        self.ROI_MIN_ANG = -95
        self.ROI_MAX_ANG = -85
        self.NORMAL_MIN_ANG = -90
        self.NORMAL_MAX_ANG = 90
        
        self.DB_EPS = 0.5 
        self.DB_MIN_SAMPLES = 5
        
        # Control
        self.REVERSE_SPEED = 60
        self.STEER_CENTER = 58  
        self.STEER_MAX_LEFT = 41  
        self.STEER_MAX_RIGHT = 79
        self.WHEELBASE = 0.55
        
        # Stage 3 Params
        self.S3_FWD_WAIT_TIME = 0.5 # [UPDATED] 1.0s delay for exit 

        # Stage 1 Config
        self.S1_STEER_A = 78        
        self.S1_STEER_B = 78         
        self.S1_CHOSEN_STEER = self.STEER_CENTER 
        
        self.S1_LIDAR_MIN = -60.0    
        self.S1_LIDAR_MAX = 45.0 # [UPDATED] Changed from 90.0 to 45.0
        self.S1_LIDAR_MAX_DIST = 5.0  
        self.S1_LIDAR_THRESH = 1.3    
        self.S1_LIDAR_THRESH = 1.3    
        self.S1_WAIT_TIME = 1.0       
        self.S1_LIDAR_DELAY = 3.0     # [NEW] Ignore Lidar for first 2.0s
        self.s1_lidar_active_logged = False # [NEW] Log flag

        # [NEW] YOLO Model Init
        self.yolo_model = None
        if YOLO is not None:
             try:
                 self.yolo_model = YOLO(MODEL_PATH)
                 print(f"[YOLO] Model loaded from {MODEL_PATH}")
             except Exception as e:
                 print(f"[Warn] YOLO Load Failed: {e}")
        
        self.homography = get_homography() # (H, H_inv)
        self.prev_lane_coeffs = None
    
    # [NEW] Reset Flags for State Switching
    def reset_flags(self):
        # self.flag_back_detect = False
        # self.flag_front_detect = False
        self.hold_start_time = 0.0
        # print("[Info] Flags Reset.")

    # [NEW] Helper for Cluster Counting
    def count_clusters_in_range(self, dist_threshold=2.0) -> int:
        if self.viz_labels is None or self.viz_points is None:
            return 0
        
        unique = set(self.viz_labels)
        if -1 in unique: unique.remove(-1)
        
        count = 0
        for k in unique:
            c_points = self.viz_points[self.viz_labels == k]
            if len(c_points) > 0:
                cx = np.mean(c_points[:, 0])
                cy = np.mean(c_points[:, 1])
                dist = math.sqrt(cx**2 + cy**2)
                if dist < dist_threshold:
                    count += 1
        return count


    # [UPDATED] Reusable YOLO Steering Logic (Returns steer, found, angle_deg)
    def get_yolo_steering(self, frame, override_offset_m=None) -> Tuple[int, bool, float]:
        """
        Returns (steering_value, is_found, lane_angle).
        steering_value: 0~100 (Default 58)
        lane_angle: Degrees w.r.t X-axis (Vertical ~ 90). None if not found.
        override_offset_m: If set, use this offset instead of CENTER_OFFSET_M
        """
        DEFAULT_STEER = self.STEER_CENTER
        
        if self.yolo_model is None:
            # print("[Error] YOLO model not loaded.")
            return DEFAULT_STEER, False, 0.0
            
        # 1. Inference
        results = self.yolo_model.predict(frame, conf=0.25, iou=0.5, verbose=False)
        r = results[0]
        
        # 2. Mask
        lane_mask = mask_from_yolo_seg(r, frame.shape, class_name=LANE_CLASS_NAME)
        if lane_mask is None:
             lane_mask = mask_from_yolo_seg(r, frame.shape, class_name=None)
        
        vis_frame = frame.copy()
        bev_dbg = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
        
        steer_deg = 0.0
        target_bev = None
        is_found = False
        calc_angle = 0.0
        
        if lane_mask is not None:
            is_found = True
            # Vis
            colored = cv.applyColorMap((lane_mask > 0).astype(np.uint8) * 255, cv.COLORMAP_JET)
            vis_frame = cv.addWeighted(vis_frame, 1.0, colored, 0.35, 0)
            
            # 3. BEV Warp
            lane_bev = bev_warp(lane_mask, self.homography[0])
            bev_dbg = cv.cvtColor(lane_bev, cv.COLOR_GRAY2BGR) # Base
            
            # 4. Curve Fit
            coeffs = fit_lane_x_of_y(lane_bev)
            if coeffs is None:
                coeffs = self.prev_lane_coeffs
            else:
                # EMA
                if self.prev_lane_coeffs is None:
                    self.prev_lane_coeffs = coeffs
                else:
                    self.prev_lane_coeffs = 0.8 * self.prev_lane_coeffs + 0.2 * coeffs
                coeffs = self.prev_lane_coeffs
                
            if coeffs is not None:
                # [NEW] Calculate Angle
                # x = ay^2 + by + c
                # Check angle near bottom (car pos)
                y_eval = float(BEV_H - 10)
                # dx/dy = 2ay + b
                dx_dy = 2 * coeffs[0] * y_eval + coeffs[1]
                
                # Angle with Horizontal X-axis:
                # Vertical line (dx/dy = 0) -> 90 deg
                # Horizontal line -> 0 deg
                # tan(theta_vert) = dx/dy
                # angle_vert = atan(dx/dy)
                # angle_horiz = 90 - angle_vert? 
                # Let's use 2 points for robustness
                y1 = float(BEV_H)
                y2 = float(BEV_H - 100)
                x1 = coeffs[0]*y1*y1 + coeffs[1]*y1 + coeffs[2]
                x2 = coeffs[0]*y2*y2 + coeffs[1]*y2 + coeffs[2]
                
                # Angle in image frame (Y down, X right)
                # We want angle with X axis.
                # dy = y1 - y2 (positive 100)
                # dx = x1 - x2 
                # angle = atan2(dy, dx)
                
                rad = math.atan2(y1 - y2, x1 - x2) 
                calc_angle = math.degrees(rad)
                # If vertical (x1=x2), angle is 90.
                
                self.lane_angle = calc_angle # Update Member Variable
                
                # 5. Center Path
                # [Modified] Use override offset if provided
                use_offset = override_offset_m if override_offset_m is not None else CENTER_OFFSET_M
                center_path = sample_center_path(coeffs, use_offset)
                
                # 6. Pure Pursuit
                steer_deg, target_bev = pure_pursuit_steer(center_path, LOOKAHEAD_M_YOLO)
                
                # Viz: Path
                overlay_path_on_bev(bev_dbg, center_path, color=(255, 0, 255))
                
                # Viz: Car Pos
                car_pos = (BEV_W//2, BEV_H-1)
                cv.circle(bev_dbg, car_pos, 10, (0,255,0), -1) 
                
                # Viz: Target & Line
                if target_bev is not None:
                     cv.circle(bev_dbg, target_bev, 8, (0,0,255), -1)
                     
                     # [Viz] Draw Predicted Path (Arc)
                     t_x, t_y = target_bev
                     c_x, c_y = car_pos
                     
                     dx = t_x - c_x
                     dy = t_y - c_y
                     dist = math.hypot(dx, dy)
                     
                     # Heading is UP (-y), Right is +x
                     # alpha = angle between Heading and Target
                     forward_dist = -dy
                     lateral_dist = dx
                     alpha = math.atan2(lateral_dist, forward_dist)
                     
                     if dist > 1.0:
                         kappa = 2.0 * math.sin(alpha) / dist
                         
                         # Simulate Arc
                         sim_x, sim_y = float(c_x), float(c_y)
                         sim_yaw = -math.pi / 2.0 # -90 deg (UP)
                         step_px = 5.0
                         
                         steps = int(dist / step_px) + 2
                         prev_p = (int(sim_x), int(sim_y))
                         
                         for _ in range(steps):
                             sim_x += step_px * math.cos(sim_yaw)
                             sim_y += step_px * math.sin(sim_yaw)
                             sim_yaw += step_px * kappa
                             
                             curr_p = (int(sim_x), int(sim_y))
                             cv.line(bev_dbg, prev_p, curr_p, (0, 255, 255), 2)
                             prev_p = curr_p
        
        # [Moved] Store frames for Main Loop display
        self.vis_frame = vis_frame
        self.bev_frame = bev_dbg
        
        servo_val = int(self.STEER_CENTER - steer_deg) 
        servo_val = max(self.STEER_MAX_LEFT, min(self.STEER_MAX_RIGHT, servo_val))
        
        return servo_val, is_found, calc_angle

    def run_stage_1(self, lidar_scan, frame=None) -> MotionCommand:
        cmd = MotionCommand(steering=self.STEER_CENTER, left_speed=0, right_speed=0)
        FWD_SPEED = 60 # Slow approach
        
        if self.state == ParkingState.S1_STRAIGHT:
            # [Integrated] Use YOLO Steering
            yolo_steer, found, _ = self.get_yolo_steering(frame)
            
            cmd.steering = yolo_steer if found else self.STEER_CENTER
            cmd.left_speed = FWD_SPEED
            cmd.right_speed = FWD_SPEED
            
            # [NEW] Ignore Lidar for first N seconds
            if time.time() - self.start_time < self.S1_LIDAR_DELAY:
                return cmd
            
            # [NEW] Print once when Lidar activates
            if not self.s1_lidar_active_logged:
                 print(f"[Stage1] Lidar ON (Delay {self.S1_LIDAR_DELAY}s Ended)")
                 self.s1_lidar_active_logged = True

            # [Maintain] Existing LiDAR Cluster Check Logic
            self.process_lidar(lidar_scan)
            
            cluster_count = 0
            if self.viz_labels is not None:
                unique = set(self.viz_labels)
                if -1 in unique: unique.remove(-1)
                cluster_count = len(unique)
                
            if cluster_count >= 1:
                print(f"[Stage1] Lidar Cluster Found (Count={cluster_count}). Stopping to Decide.")
                self.state = ParkingState.S1_DECIDE
                
        elif self.state == ParkingState.S1_DECIDE:
            # Stop & Decide
            self.process_lidar(lidar_scan)
            min_dist = 999.0
            
            if self.viz_labels is not None and self.viz_points is not None:
                unique = set(self.viz_labels)
                if -1 in unique: unique.remove(-1)
                
                for k in unique:
                    c_points = self.viz_points[self.viz_labels == k]
                    if len(c_points) > 0:
                        cx = np.mean(c_points[:, 0])
                        cy = np.mean(c_points[:, 1])
                        dist = math.sqrt(cx**2 + cy**2)
                        if dist < min_dist:
                            min_dist = dist
            
            self.viz_s1_min_dist = min_dist 
            print(f"[Stage1] Closest Cluster Dist: {min_dist:.2f}m")
            
            if min_dist < self.S1_LIDAR_THRESH:
                self.S1_CHOSEN_STEER = self.S1_STEER_A
                print(f"[Stage1] Close ({min_dist:.2f} < {self.S1_LIDAR_THRESH}). Steer A ({self.S1_STEER_A})")
            else:
                self.S1_CHOSEN_STEER = self.S1_STEER_B
                print(f"[Stage1] Far ({min_dist:.2f} >= {self.S1_LIDAR_THRESH}). Steer B ({self.S1_STEER_B})")
            
            self.ROI_MIN_ANG = self.S1_LIDAR_MIN
            self.ROI_MAX_ANG = self.S1_LIDAR_MAX
            self.LIDAR_MAX_DIST = self.S1_LIDAR_MAX_DIST 
            
            self.state = ParkingState.S1_CURVE
            
        elif self.state == ParkingState.S1_CURVE:
            cmd.steering = self.S1_CHOSEN_STEER
            cmd.left_speed = FWD_SPEED
            cmd.right_speed = FWD_SPEED
            
            target = self.process_lidar(lidar_scan)
            if target:
                print(f"[Stage1] Clusters Found. Transitioning to Stage 2.")
                self.state = ParkingState.S1_WAIT
                self.state_timer = time.time()
                
        elif self.state == ParkingState.S1_WAIT:
            cmd.steering = self.STEER_CENTER
            cmd.left_speed = 0
            cmd.right_speed = 0
            
            if time.time() - self.state_timer > self.S1_WAIT_TIME:
                print("[Stage1] Wait Complete. Starting Stage 2 (REVERSE).")
                self.ROI_MIN_ANG = self.NORMAL_MIN_ANG # Full Range for Parking
                self.ROI_MAX_ANG = self.NORMAL_MAX_ANG
                self.LIDAR_MAX_DIST = self.NORMAL_MAX_DIST 
                self.reset_flags()
                self.state = ParkingState.REVERSE 
                
        return cmd

    # [NEW] Separated Stage 3 Logic
    def run_stage_3(self, frame=None) -> MotionCommand:
        cmd = MotionCommand(steering=self.STEER_CENTER, left_speed=0, right_speed=0)
        FWD_SPEED = 100 
        
        if self.state == ParkingState.S3_FWD_CHECK:
            # [UPDATED] Skip Ultrasonic Check -> Blind Forward 1s
            print(f"[Stage3] StartBlind Forward {self.S3_FWD_WAIT_TIME}s...")
            self.state = ParkingState.S3_FWD_WAIT
            self.state_timer = time.time()
                 
        elif self.state == ParkingState.S3_FWD_WAIT:
            # Continue Forward (Existing Logic)
            cmd.steering = self.STEER_CENTER
            cmd.left_speed = FWD_SPEED
            cmd.right_speed = FWD_SPEED
            
            if time.time() - self.state_timer > self.S3_FWD_WAIT_TIME:
                 print(f"[Stage3] {self.S3_FWD_WAIT_TIME}s Wait Done. Turning Right.")
                 self.state = ParkingState.S3_RIGHT_TURN
                 
        elif self.state == ParkingState.S3_RIGHT_TURN:
            # [Refine] Try Pure Pursuit for Turn, Fallback to Fixed
            yolo_steer, found, angle = self.get_yolo_steering(frame)
            
            if found:
                # Use Pure Pursuit for optimal turning
                cmd.steering = yolo_steer
                
                # [Debug] Print Angle
                print(f"[Stage3-Turn] Angle: {angle:.1f} deg")
                
                # Check Angle (Transition Condition)
                # If angle is vertical enough > 65 deg
                if abs(angle) < 95.0:
                     print(f"[Stage3] Angle Good ({angle:.1f} > 50). Go Straight.")
                     self.state = ParkingState.S3_STRAIGHT
            else:
                # Fallback: Hard Right Turn (Blind)
                cmd.steering = 41
                
            cmd.left_speed = FWD_SPEED
            cmd.right_speed = FWD_SPEED
            
            if self.lane_angle is not None:
                if abs(self.lane_angle) >= self.lane_angle_threshold:
                     pass # Already handled by steering check above or keep as backup?

        elif self.state == ParkingState.S3_STRAIGHT:
             # User Request: Lock steering to straight (No adjustments)
             cmd.steering = self.STEER_CENTER # 58
             cmd.left_speed = FWD_SPEED
             cmd.right_speed = FWD_SPEED
             
        return cmd

    # [Renamed] Scan Gap Logic (Existing)
    def run_scan_gap(self, lidar_scan) -> MotionCommand:
        cmd = MotionCommand(steering=self.STEER_CENTER, left_speed=0, right_speed=0)
        
        target = self.process_lidar(lidar_scan)
        if target:
            self.target_point = target
            print(f"[Logic] Target Found at {target}. Switching to REVERSE.")
            self.reset_flags()
            self.state = ParkingState.REVERSE
        else:
            # Drive slowly forward to scan? (Currently pass/stop based on existing code)
            pass
            
        return cmd

    # [Renamed] Reverse Parking Logic (Existing)
    def run_reverse_parking(self, lidar_scan) -> MotionCommand:
        cmd = MotionCommand(steering=self.STEER_CENTER, left_speed=0, right_speed=0)
        
        # 0. Wait Hold Logic
        if self.state == ParkingState.WAIT_HOLD:
             if time.time() - self.hold_start_time > 5.0:
                 print("[Logic] Hold Complete. Starting Stage 3 (Exit).")
                 self.state = ParkingState.S3_FWD_CHECK
             return cmd

        if self.state == ParkingState.FINISHED:
             # Wait for transition to Stage 3 or just stop
             return cmd

        current_target = self.process_lidar(lidar_scan)
        if current_target:
            self.target_point = current_target
        
        if self.target_point:
             tx, ty = self.target_point
             dist = math.sqrt(tx**2 + ty**2)
             
             if dist < 0.3:
                 print("[Logic] Reached Target. FINISHED.")
                 self.state = ParkingState.FINISHED
                 return cmd

             steer, speed = self.calculate_control(self.target_point)
             
             # [Alignment Check]
             is_aligned = False
             if self.viz_center_line:
                 a, b, c = self.viz_center_line
                 vx, vy = b, -a
                 if vx < 0: vx, vy = -vx, -vy 
                 angle_rad = math.atan2(vy, vx)
                 angle_deg = math.degrees(angle_rad)
                 if abs(angle_deg) < 3.0:
                     print(f"[Logic] Aligned ({angle_deg:.1f} deg). Forcing Straight.")
                     steer = self.STEER_CENTER
                     is_aligned = True
             
             cmd.steering = steer
             cmd.left_speed = speed
             cmd.right_speed = speed
             
             # [Ultrasonic Safety Check] Generally Check (User request: Don't wait for align)
             # if True: # was if is_aligned:
             #     # d1=fl, d2=fr, d3=br
             #     fl, fr, br = self.us_sensors
             #    #  print(f"[Debug] Reverse US: {fl}, {fr}, {br}")
                 
             #     # 2) front_detect (fl < 100 and fr < 100)
             #     # User Request: Stop if two sensors < 100
             #     # [Fix] Add lower bound to prevent 0 noise (if 0 means error)
             #     if (1 < fl < 80) and (1 < fr < 80):
             #        self.flag_front_detect = True
             #        print(f"[Debug] Both Front Sensors < 70cm! fl={fl}, fr={fr}")
             #        print(f"[Logic] Flag: Front Detected ({fl},{fr}cm)")

             #         self.state = ParkingState.WAIT_HOLD
             #         self.hold_start_time = time.time()
             #         return MotionCommand(steering=self.STEER_CENTER, left_speed=0, right_speed=0)

        # [NEW Logic] 2-Stage Stop Check
        # 1. Trigger: If 2 clusters < 1.0m -> Enter CHECK_STOP mode
        if self.state == ParkingState.REVERSE:
             cnt = self.count_clusters_in_range(1.0)
             if cnt >= 2:
                 print(f"[Logic] 2 Clusters Detected < 1.0m (cnt={cnt}). Checking for Stop...")
                 self.state = ParkingState.REVERSE_CHECK_STOP
                 
        # 2. Stop: If 0 clusters < 2.0m -> STOP
        elif self.state == ParkingState.REVERSE_CHECK_STOP:
             cnt = self.count_clusters_in_range(2.0)
             if cnt == 0:
                 print(f"[Logic] All Clusters Cleared (cnt={cnt}). STOPPING.")
                 self.state = ParkingState.WAIT_HOLD
                 self.hold_start_time = time.time()
                 return MotionCommand(steering=self.STEER_CENTER, left_speed=0, right_speed=0)
             else:
                 # Still moving...
                 # print(f"[Debug] Waiting for clear... cnt={cnt}")
                 pass

        return cmd

        return cmd

    def run_step(self, lidar_scan: Tuple[np.ndarray, np.ndarray], frame=None) -> MotionCommand:
        # [NEW] Input Mode Dispatch
        if self.state == ParkingState.INPUT_MODE:
             return self.run_input_mode(frame)

        # 1. State Dispatcher
        
        # Stage 1: Approach & Positioning
        if self.state in [ParkingState.S1_STRAIGHT, ParkingState.S1_DECIDE, 
                          ParkingState.S1_CURVE, ParkingState.S1_WAIT]:
            return self.run_stage_1(lidar_scan, frame)

        # Stage 3: Exit Parking (User Implementation)
        if self.state in [ParkingState.S3_FWD_CHECK, ParkingState.S3_FWD_WAIT, 
                          ParkingState.S3_RIGHT_TURN, ParkingState.S3_STRAIGHT]:
            return self.run_stage_3(frame)

        # Reverse Parking
        elif self.state in [ParkingState.REVERSE, ParkingState.REVERSE_CHECK_STOP, 
                            ParkingState.WAIT_HOLD, ParkingState.FINISHED]:
            return self.run_reverse_parking(lidar_scan)

        # Scan Gap
        elif self.state == ParkingState.SCAN_GAP:
            return self.run_scan_gap(lidar_scan)
            
        return MotionCommand(steering=self.STEER_CENTER, left_speed=0, right_speed=0)

    def process_lidar(self, lidar_scan: Tuple[np.ndarray, np.ndarray]) -> Optional[Tuple[float, float]]:
        ranges, angles = lidar_scan
        if len(ranges) == 0: return None
            
        points = scan_to_points(angles, ranges, 
                                min_dist=0.1, max_dist=self.LIDAR_MAX_DIST,
                                min_angle=self.ROI_MIN_ANG, max_angle=self.ROI_MAX_ANG)
        
        # Save for viz
        self.viz_points = None
        self.viz_labels = None
        self.viz_corners = []
        self.viz_center_line = None
        self.viz_target = None
        self.viz_arc_path = None
        
        if len(points) < 5: return None
        
        # [Offset Correction] Shift points to Vehicle Frame (Center)
        # Lidar is at x = -0.4. Point at x_lidar = 1.0 -> x_vehicle = 0.6
        points[:, 0] += LIDAR_X_OFFSET

        # Filter for Right Side

        right_mask = points[:, 1] < 1.0 
        points = points[right_mask]
        
        if len(points) < 5: return None
        
        labels = dbscan(points, eps=self.DB_EPS, min_samples=self.DB_MIN_SAMPLES)
        
        # Save viz data
        self.viz_points = points
        self.viz_labels = labels
        
        clusters = []
        unique_labels = [k for k in np.unique(labels) if k != -1]
        for k in unique_labels:
            clusters.append(points[labels == k])
            
        clusters.sort(key=len, reverse=True)
        
        target_res = None
        
        if len(clusters) >= 2:
            c1 = clusters[0]
            c2 = clusters[1]
            p1, p2, dist = find_closest_pair_between_clusters(c1, c2)
            
            if p1 is not None:
                self.viz_corners = [p1, p2]
                res = get_center_line_from_corners(p1, p2)
                if res:
                    line, mid = res
                    self.viz_center_line = line
                    
                    # Sliding Target Logic
                    # Always look 1.0m ahead of specific projection on the line
                    lookahead = 1.0 
                    target = calculate_sliding_target(line, (0.0, 0.0), lookahead)

                    # dist_to_entry = math.sqrt(mid[0]**2 + mid[1]**2)
                    # lookahead = 1.0  # User Request: Fixed lookahead
                    # # lookahead = 1.0 if dist_to_entry < 1.0 else 0.0
                    # 
                    # if lookahead > 0:
                    #     target = calculate_pure_pursuit_target(line, mid, lookahead)
                    # else:
                    #     target = mid
                        
                    self.viz_target = target
                    target_res = (target[0], target[1])
                    
                    # Generate Arc Path Visualization
                    self.viz_arc_path = generate_arc_path((0.0, 0.0), target)

        return target_res

    def calculate_control(self, target_point) -> Tuple[int, int]:

        gx, gy = target_point
        ld = math.sqrt(gx**2 + gy**2) 
        if ld < 0.1: return self.STEER_CENTER, 0 
            
        alpha = math.atan2(gy, gx) 
        
        delta = math.atan((2 * self.WHEELBASE * math.sin(alpha)) / ld)
        delta_deg = math.degrees(delta)
        
        steer = self.STEER_CENTER + int(delta_deg * 3) # 좀 더 빨리 회전하고 싶으면 1.5를 더 크게
        steer = max(self.STEER_MAX_LEFT, min(self.STEER_MAX_RIGHT, steer))
        
        return steer, -self.REVERSE_SPEED

    def update_plot(self, ax, cmd: MotionCommand):
        ax.clear()
        
        # 1. Points
        if self.viz_points is not None and self.viz_labels is not None:
            unique_draw = np.unique(self.viz_labels)
            for k in unique_draw:
                if k == -1:
                    m = self.viz_labels == -1
                    ax.scatter(self.viz_points[m,0], self.viz_points[m,1], s=2, c='gray')
                else:
                    m = self.viz_labels == k
                    ax.scatter(self.viz_points[m,0], self.viz_points[m,1], s=10)

        # 2. Corners
        if len(self.viz_corners) == 2:
             p1, p2 = self.viz_corners
             ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=1)
             ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=80, c='red', marker='*')
             
        # 3. Center Line
        if self.viz_center_line is not None:
             draw_line(ax, self.viz_center_line, xlim=(0, 2.0), color='purple', linestyle='--', linewidth=2)
             
        # 4. Target
        if self.viz_target is not None:
             ax.scatter([self.viz_target[0]], [self.viz_target[1]], s=100, c='blue', marker='x')
             
        # 5. Arc Path
        if self.viz_arc_path is not None:
             ax.plot(self.viz_arc_path[:,0], self.viz_arc_path[:,1], 'g-', linewidth=2)
             
        # 6. Ego & Lidar
        ax.scatter([0], [0], s=60, c='black', label='Ego Center')
        ax.arrow(0, 0, 0.2, 0, head_width=0.05, fc='k')
         
        # Draw Lidar pos
        ax.scatter([LIDAR_X_OFFSET], [0], s=30, c='cyan', marker='s', label='Lidar')

        # 6. Basic Settings
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-4.0, 4.0)
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        
        title_str = f"State: {self.state.name} | Steer: {cmd.steering} | Speed: {cmd.left_speed}"
        # [NEW] Add Angle Info
        if self.lane_angle is not None:
             title_str += f" | Angle: {self.lane_angle:.1f}"
             
        # [NEW] Add S1 Distance Info
        if self.viz_s1_min_dist is not None:
             title_str += f" | S1 Dist: {self.viz_s1_min_dist:.2f}m"

        ax.set_title(title_str)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lidar-port', type=str, default=LIDAR_PORT)
    parser.add_argument('--arduino-port', type=str, default=ARDUINO_PORT)
    parser.add_argument('--real-lidar', action='store_true', default=True)
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--stage3', action='store_true', help="Start directly in Stage 3 (Exit Parking)")
    parser.add_argument('--s3-straight', action='store_true', help="Start directly in Stage 3 Straight Mode") 
    parser.add_argument('--s3-turn', action='store_true', help="Start directly in Stage 3 Right Turn") # [NEW]
    parser.add_argument('--reverse', action='store_true', help="Start directly in Reverse Mode")
    parser.add_argument('--reverse-check', action='store_true', help="Start directly in Reverse Check Stop Mode") # [NEW]
    parser.add_argument('--s1-curve', action='store_true', help="Start directly in Stage 1 Curve") # [NEW]
    parser.add_argument('--s1-wait', action='store_true', help="Start directly in Stage 1 Wait") # [NEW]
    parser.add_argument('--cam', type=int, default=1, help="Camera Index for Lane Detection")
    args = parser.parse_args()

    print(f"[Main] Lidar Parking (Matplotlib Viz). Real={args.real_lidar} Stage3={args.stage3}")
    
    # [NEW] Camera Init
    cap = None
    
    try:
        # Try to open camera with DirectShow (Better for Windows)
        cap = cv.VideoCapture(args.cam, cv.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[Warn] CAP_DSHOW failed for Cam {args.cam}. Trying default...")
            cap = cv.VideoCapture(args.cam)
            
        if not cap.isOpened():
            print(f"[Warn] Camera {args.cam} not found. Vision disabled.")
            cap = None
        else:
            print(f"[Vision] Camera {args.cam} Opened.")
            # Set resolution if needed? 
            # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
            # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
                
    except Exception as e:
        print(f"[Warn] Vision Init Fail: {e}")

    lidar_gen = None
    lidar_hw = None
    if args.real_lidar:
        try:
             lidar_hw = libLidar(args.lidar_port)
             lidar_hw.init()
             lidar_gen = lidar_hw.scanning()
             print(f"[LiDAR] Connected on {args.lidar_port}.")
        except Exception as e:
             print(f"[Error] Lidar Fail: {e}")
             sys.exit(1)
             
    ser = None
    if args.real_lidar:
        try:
            ser = SerialSender(SerialConfig(port=args.arduino_port, baud=args.baud))
            print(f"[Serial] Opened on {args.arduino_port}.")
        except Exception as e:
            print(f"[Warn] Serial Fail: {e}")

    parking = VerticalParkingLidar()
    if args.s3_straight:
         print("[Info] Starting in Stage 3 STRAIGHT Mode.")
         parking.state = ParkingState.S3_STRAIGHT
    elif args.s3_turn:
         print("[Info] Starting in Stage 3 RIGHT TURN Mode.")
         parking.state = ParkingState.S3_RIGHT_TURN
    elif args.stage3:
        print("[Info] Starting in Stage 3 Mode (FWD_CHECK).")
        parking.state = ParkingState.S3_FWD_CHECK
    elif args.reverse_check:
        print("[Info] Starting in Reverse Check Stop Mode.")
        parking.state = ParkingState.REVERSE_CHECK_STOP
    elif args.reverse:
        print("[Info] Starting in Reverse Mode.")
        parking.state = ParkingState.REVERSE
    elif args.s1_curve:
        print("[Info] Starting in Stage 1 Curve Mode.")
        parking.state = ParkingState.S1_CURVE
        # Note: Might need to set initial steering if strictly testing curve
    elif args.s1_wait:
        print("[Info] Starting in Stage 1 Wait Mode.")
        parking.state = ParkingState.S1_WAIT
        parking.state_timer = time.time()
    else:
        print("[Info] Starting in Normal Parking Mode (S1_STRAIGHT).")
        # Default is S1_STRAIGHT as set in __init__
        pass
    
    # Init Plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Key Event Handler
    running_status = {'run': True, 'active': False} # [NEW] Driving Active Flag
    def on_key(event):
        if event.key == 'q':
            running_status['run'] = False
        elif event.key == ' ': # [NEW] Spacebar Toggle
            running_status['active'] = not running_status['active']
            if running_status['active']:
                parking.start_time = time.time()
                parking.s1_lidar_active_logged = False # [NEW] Reset log flag
            print(f"[System] Driving Active: {running_status['active']}")
        
        # [Debug] State Switching Keys
        elif event.key == '1':
            print("[Debug] Request State -> SCAN_GAP")
            parking.state = ParkingState.SCAN_GAP
        elif event.key == '2':
            print("[Debug] Request State -> REVERSE")
            parking.reset_flags()
            parking.state = ParkingState.REVERSE
        elif event.key == '3':
            print("[Debug] Request State -> S3_FWD_CHECK")
            parking.state = ParkingState.S3_FWD_CHECK
        elif event.key == '4':
            print("[Debug] Request State -> S3_RIGHT_TURN")
            parking.state = ParkingState.S3_RIGHT_TURN
        elif event.key == '0':
            print("[Debug] Request State -> FINISHED")
            parking.state = ParkingState.FINISHED
            
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\n[DEBUG CONTROLS]")
    print(" 'q': Quit")
    print(" '1': Force State SCAN_GAP")
    print(" '2': Force State REVERSE")
    print(" '3': Force State S3_FWD_CHECK")
    print(" '4': Force State S3_RIGHT_TURN")
    print(" '0': Force State FINISHED\n")
    
    last_send_t = 0
    SEND_HZ = 20.0
    
    try:
        while running_status['run']:
            # 0. Vision Processing
            frame = None
            if cap is not None and cap.isOpened():
                ret, img = cap.read()
                if ret:
                    frame = img
                    # Note: process_vision_control draws on 'frame' in-place if called.
                    # But run_step calls it only in specific states.
                    # If we want to verify vision ALWAYS, we might want to run it?
                    # But that affects control logic.
                    # For now, just show the frame. run_step will modify it if vision is active.
                    
                    pass
                    # cv.imshow("Vision Control", frame) # Moved down
                    # cv.waitKey(1)
                else:
                    pass
            
            # Data Acquisition
            scan_data = (np.array([]), np.array([]))
            if args.real_lidar and lidar_gen:
                try:
                    raw = next(lidar_gen)
                    if len(raw) > 0:
                        deg = raw[:, 0]
                        dist = raw[:, 1]
                        valid = dist > 0
                        scan_data = (dist[valid]/1000.0, np.deg2rad(deg[valid]))
                except: pass
            else:
                # SIMULATION
                sim_r, sim_a = [], []
                for deg in range(0, 360):
                    a = np.deg2rad(deg)
                    if np.sin(a) < -0.2:
                        d_wall = -1.5 / np.sin(a)
                        x_wall = d_wall * np.cos(a)
                        if -1.0 < x_wall < 1.5: d = 6.0
                        else: d = d_wall
                    else: d = 5.0
                    if 0 < d < 8:
                        noise = np.random.uniform(-0.02, 0.02)
                        sim_r.append(d + noise)
                        sim_a.append(a)
                scan_data = (np.array(sim_r), np.array(sim_a))

            # Logic Step
            parking.vis_frame = None # Reset shared frame
            parking.bev_frame = None
            cmd = parking.run_step(scan_data, frame=frame)

            # [Moved] Show Vision after processing (to see overlays)
            if frame is not None:
                # Decide which frame to show
                display_frame = parking.vis_frame if parking.vis_frame is not None else frame
                
                # [NEW] Draw Status
                status_str = "Press SPACE to Stop" if running_status['active'] else "Press SPACE to Start"
                status_color = (0, 255, 0) if running_status['active'] else (0, 0, 255)
                # Draw on display_frame (might be a copy or raw)
                # Ensure we are drawing on a writeable array
                cv.putText(display_frame, status_str, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                
                cv.imshow("Vision Control", display_frame)
                
                # Show BEV if available
                if parking.bev_frame is not None:
                     cv.imshow("BEV View", parking.bev_frame)
                
                key = cv.waitKey(1)
                if key == ord('q'):
                    running_status['run'] = False
                elif key == ord(' '):
                    running_status['active'] = not running_status['active']
                    if running_status['active']:
                        parking.start_time = time.time() # [NEW] Reset timer on start
                    print(f"[System] Driving Active: {running_status['active']}")
            
            # Serial Send and Read (Parse US)
            if ser:
                now = time.time()
                if (now - last_send_t) > (1.0/SEND_HZ):
                    # [NEW] Gate Command based on Active Flag
                    if not running_status['active']:
                        # Force Stop
                        cmd.steering = parking.STEER_CENTER
                        cmd.left_speed = 0
                        cmd.right_speed = 0
                    
                    ser.send(cmd)
                    last_send_t = now
                
                # [NEW] Read Serial for Ultrasonic Data - REMOVED
                # raw_data = ser.read()



            # Matplotlib Update
            parking.update_plot(ax, cmd)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # In sim mode, maybe sleep a bit to not burn CPU?
            # But real lidar blocks on 'next(lidar_gen)'.
            if not args.real_lidar:
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        pass
    finally:
        if ser:
            ser.send(MotionCommand(60, 0, 0))
            ser.close()
        if lidar_hw:
            lidar_hw.stop()
        if cap:
            cap.release()
        cv.destroyAllWindows()
        plt.close()
        plt.close()
