import cv2
import time
import sys
import numpy as np

# Try to import win_autodrive modules
try:
    from win_autodrive.messages import MotionCommand
    from win_autodrive.serial_io import SerialConfig, SerialSender
except ImportError:
    # Extend path if needed (assuming common structure)
    sys.path.append("c:/ggdrive/parking")
    try:
        from win_autodrive.messages import MotionCommand
        from win_autodrive.serial_io import SerialConfig, SerialSender
    except ImportError:
        print("Error: Could not import win_autodrive modules.")
        print("Please ensure you are in the correct directory or the environment is set up.")
        sys.exit(1)

def main():
    # ==========================================================================
    # Configuration
    # ==========================================================================
    PORT = 'COM9'      # Default Arduino Port
    BAUD = 115200
    
    # Vehicle Config from 3_parking.py
    STEER_CENTER = 58
    STEER_MAX_LEFT = 41
    STEER_MAX_RIGHT = 79
    STEER_STEP = 5      # Amount to change steering per key press
    
    SPEED_FWD = 60
    SPEED_REV = -60
    
    # ==========================================================================
    # Initialization
    # ==========================================================================
    print(f"Attempting to connect to Arduino on {PORT}...")
    ser = None
    try:
        ser = SerialSender(SerialConfig(port=PORT, baud=BAUD))
        print(f"Success! Connected to {PORT}.")
        print("Waiting 2s for Arduino reset...")
        time.sleep(2.0) # [NEW] Essential for Arduino Nano/Uno reset
    except Exception as e:
        print(f"Connection Failed: {e}")
        print("Running in dummy mode (Visualization only).")
        ser = None

    # Initial State
    current_steer = STEER_CENTER
    current_speed = 0
    
    # Instructions
    print("-" * 50)
    print("KEYBOARD CONTROL MODE")
    print("-" * 50)
    print("  W      : Forward (Speed 60)")
    print("  S      : Reverse (Speed -60)")
    print("  SPACE  : Stop (Speed 0)")
    print("  A      : Steer Left")
    print("  D      : Steer Right")
    print("  R      : Reset Steering to Center")
    print("  Q      : Quit")
    print("-" * 50)

    # Create Display Window
    window_name = "Arduino Keyboard Control"
    cv2.namedWindow(window_name)
    img = np.zeros((300, 500, 3), dtype=np.uint8)

    try:
        while True:
            # 1. Visualization
            img[:] = 20  # Dark background
            
            # Draw Values
            cv2.putText(img, f"Speed: {current_speed}", (30, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(img, f"Steer: {current_steer}", (30, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Visual Bar for Steer
            bar_width = 300
            bar_start_x = 100
            bar_y = 180
            cv2.rectangle(img, (bar_start_x, bar_y), (bar_start_x + bar_width, bar_y + 20), (100, 100, 100), 2)
            
            # Map steer to x position
            # range 41..79 -> 0..1 (approx 38 range)
            norm_steer = (current_steer - STEER_MAX_LEFT) / (STEER_MAX_RIGHT - STEER_MAX_LEFT)
            norm_steer = max(0.0, min(1.0, norm_steer))
            indicator_x = int(bar_start_x + norm_steer * bar_width)
            cv2.circle(img, (indicator_x, bar_y + 10), 8, (0, 255, 255), -1)
            
            # Status Text
            status_text = "Stopped"
            color = (100, 100, 100)
            if current_speed > 0: 
                status_text = "FORWARD"
                color = (0, 255, 0)
            elif current_speed < 0: 
                status_text = "REVERSE"
                color = (0, 0, 255)
            
            cv2.putText(img, status_text, (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            cv2.imshow(window_name, img)
            
            # 2. Input Handling
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            
            elif key == ord('w'):
                current_speed = SPEED_FWD
            elif key == ord('s'):
                current_speed = SPEED_REV
            elif key == ord(' '):
                current_speed = 0
            
            elif key == ord('a'):
                current_steer -= STEER_STEP
                if current_steer < STEER_MAX_LEFT: 
                    current_steer = STEER_MAX_LEFT
            elif key == ord('d'):
                current_steer += STEER_STEP
                if current_steer > STEER_MAX_RIGHT: 
                    current_steer = STEER_MAX_RIGHT
            elif key == ord('r'):
                current_steer = STEER_CENTER

            # 3. Send Command
            if ser:
                cmd = MotionCommand(
                    steering=current_steer, 
                    left_speed=current_speed, 
                    right_speed=current_speed
                )
                ser.send(cmd)

    except KeyboardInterrupt:
        print("\nInterrupted by User")
        
    finally:
        # Stop Vehicle on Exit
        if ser:
            print("Stopping Vehicle...")
            try:
                ser.send(MotionCommand(steering=STEER_CENTER, left_speed=0, right_speed=0))
                time.sleep(0.5)
                ser.close()
            except:
                pass
        
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
