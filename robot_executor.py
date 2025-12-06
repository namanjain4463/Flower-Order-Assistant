#!/usr/bin/env python3
# ------------------------------------------------------------
# UR5 Smart Picker - Standalone Execution Script
# ------------------------------------------------------------
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import socket, time, math, os, cv2, sys, json
import numpy as np
import pandas as pd
from gpiozero import Servo
from time import sleep
from datetime import datetime
from ultralytics import YOLO
import torch
import torch.nn as nn
from pathlib import Path
import io 

#torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

# ============================================================
# 1. CONFIGURATION
# ============================================================
UR_IP = "169.254.152.222"
# Use the clean model file
MODEL_PATH = Path("flower_joint_model_CLEAN.pth") 
# Use the renamed YOLO file
YOLO_PATH = "best_yolo_CLEAN.pt" 
IMG_W = 1280
IMG_H = 720

# Positions
P_CAM  = [math.radians(x) for x in [-61.24, -89.61, 50.15, -57.46, -92.83, 115.69]]

# 5 Drop Locations
DROP_POSITIONS = {
    'A': [math.radians(x) for x in [80.52, -75.77, 88.12, -101.97, -93.01, 77.96]],
    'B': [math.radians(x) for x in [70.00, -75.77, 88.12, -101.97, -93.01, 77.96]],
    'C': [math.radians(x) for x in [60.00, -85.00, 88.12, -101.97, -93.01, 77.96]],
    'D': [math.radians(x) for x in [50.00, -90.00, 95.00, -101.97, -93.01, 77.96]],
    'E': [math.radians(x) for x in [90.00, -70.00, 80.00, -101.97, -93.01, 77.96]]
}

# Hardware
try:
    # Assuming this runs on a Raspberry Pi or similar
    servo = Servo(12, min_pulse_width=0.0005, max_pulse_width=0.0025)
except Exception:
    class DummyServo:
        def min(self): pass
        def max(self): pass
        def mid(self): pass
        def value(self, val): pass
        def close(self): pass
    servo = DummyServo()

# ============================================================
# 2. MODEL LOADING
# ============================================================
class YOLOtoJointMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Linear(32, 128), nn.Dropout(0.3),
                                 nn.ReLU(), nn.Linear(128, 16), nn.Linear(16, 6))
    def forward(self, x): return self.net(x)

def load_mlp(path):
    # Robust loading using BytesIO to prevent path path conflicts
    path_str = str(path)
    with open(path_str, 'rb') as f:
        buffer = io.BytesIO(f.read())
        
    # Attempt to load the checkpoint file
    ckpt = torch.load(buffer, map_location="cpu", weights_only=False) 
    
    # --- Start of Model Loading Fix ---
    # Extract the state_dict, prioritizing 'model' key, then the root dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        # Fallback if the dict keys are the state_dict keys directly (less common but possible)
        state_dict = ckpt
    else:
        # If ckpt is not a dictionary (e.g., a pickled model instance), 
        # this path often leads to the original error.
        raise ValueError("Invalid checkpoint structure: Expected a dictionary containing state_dict.")

    m = YOLOtoJointMLP()
    m.load_state_dict(state_dict) 
    m.eval()
    
    # Ensure scalers are present before returning
    if not all(key in ckpt for key in ["x_scaler_mean", "x_scaler_scale", "y_scaler_mean", "y_scaler_scale"]):
        raise KeyError("Missing required scaler keys in the model checkpoint.")

    return (m, 
            torch.tensor(ckpt["x_scaler_mean"], dtype=torch.float32), 
            torch.tensor(ckpt["x_scaler_scale"], dtype=torch.float32),
            torch.tensor(ckpt["y_scaler_mean"], dtype=torch.float32), 
            torch.tensor(ckpt["y_scaler_scale"], dtype=torch.float32))
    # --- End of Model Loading Fix ---

# ============================================================
# 3. UTILITIES
# ============================================================
class KNNErrorCompensator:
    def __init__(self, csv_path="xyz_error_log.csv", k=3):
        self.csv_path = csv_path
        self.k = k
        self.data = pd.DataFrame(columns=["feat_x", "feat_y", "diff_x", "diff_y", "diff_z"])
        if os.path.exists(self.csv_path):
            try:
                loaded = pd.read_csv(self.csv_path)
                if not loaded.empty: self.data = loaded
            except: pass

    def get_correction(self, fx, fy):
        if len(self.data) == 0: return [0.0, 0.0, 0.0]
        dists = np.sqrt((self.data["feat_x"] - fx)**2 + (self.data["feat_y"] - fy)**2)
        idx = dists.nsmallest(min(self.k, len(self.data))).index
        return self.data.loc[idx, ["diff_x", "diff_y", "diff_z"]].mean().values

    def log_error(self, fx, fy, p_before, p_after):
        diff = [p_after[i] - p_before[i] for i in range(3)]
        entry = pd.DataFrame([{"feat_x": fx, "feat_y": fy, "diff_x": diff[0], "diff_y": diff[1], "diff_z": diff[2]}])
        self.data = pd.concat([self.data, entry], ignore_index=True)
        self.data.to_csv(self.csv_path, index=False)

def predict_joints(val, m, xm, xs, ym, ys):
    with torch.no_grad(): 
        t = (torch.tensor(val, dtype=torch.float32) - xm) / xs
        return (m(t) * ys + ym).numpy()

def robot_startup():
    with socket.create_connection((UR_IP, 29999), timeout=2) as s:
        s.recv(1024); s.sendall(b"play\n"); time.sleep(0.5)

def capture_img():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, IMG_W); cap.set(4, IMG_H); time.sleep(0.5)
    ret, frame = cap.read(); cap.release()
    if ret:
        path = f"snap_{datetime.now().strftime('%H%M%S')}.jpg"
        cv2.imwrite(path, frame)
        return path
    return None

def draw_and_show_target(img_path, all_detections, target_index):
    img = cv2.imread(img_path)
    if img is None: return
    W, H = img.shape[1], img.shape[0]
    for idx, det in enumerate(all_detections):
        xn, yn, wn, hn = det['box_norm']
        color_name = det['color']
        x_min = int((xn - wn / 2) * W); y_min = int((yn - hn / 2) * H)
        x_max = int((xn + wn / 2) * W); y_max = int((yn + hn / 2) * H)
        color = (0, 255, 0) if idx == target_index else (0, 0, 255)
        text = f"TARGET: {color_name}" if idx == target_index else f"{color_name}"
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2 if idx!=target_index else 4)
        cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    # Using waitKey(1) ensures this runs smoothly in a script loop
    cv2.imshow("Robot Target View", img); cv2.waitKey(1)

# ============================================================
# 4. EXECUTION LOGIC
# ============================================================
def execute_flower_order(bom_list):
    if not bom_list: return "Empty BOM.", []

    pick_plan = []
    for item in bom_list:
        dropoff_key = item.get('dropoffLocation', 'A').upper()
        if dropoff_key not in DROP_POSITIONS: dropoff_key = 'A'
        for _ in range(item['quantity']):
            pick_plan.append({'color': item['color'], 'dropoff': dropoff_key})
            
    unavailable_items = []
    results = []
    knn = KNNErrorCompensator()
    
    try:
        print("[ROBOT] Loading Models...")
        mlp, xm, xs, ym, ys = load_mlp(MODEL_PATH)
        yolo_model = YOLO(YOLO_PATH)
        print("[ROBOT] Models Ready.")
        
        robot_startup()
        rtde_c = RTDEControl(UR_IP)
        rtde_r = RTDEReceive(UR_IP)

        for idx, target_item in enumerate(pick_plan):
            target_color = target_item['color']
            dropoff_loc_key = target_item['dropoff']
            target_drop_pos = DROP_POSITIONS[dropoff_loc_key]

            print(f"\n[Cycle #{idx+1}] Picking {target_color} -> {dropoff_loc_key}")
            
            rtde_c.moveJ(P_CAM, speed=0.8, acceleration=0.5)
            servo.min(); sleep(0.5)

            img_path = capture_img()
            if not img_path: 
                results.append(f"Camera error: {target_color}")
                unavailable_items.append(target_item)
                continue
            
            res = yolo_model.predict(img_path, conf=0.25)[0]
            all_detections = [] 
            valid_targets = []
            
            for box in res.boxes:
                xn, yn, wn, hn = box.xywhn[0].tolist()
                color_name = res.names[int(box.cls)]
                kamus = {
                    "prediction": predict_joints([[yn]], mlp, xm, xs, ym, ys),
                    "position": xn, "y_pos": yn, "box_norm": [xn, yn, wn, hn],
                    "color": color_name
                }
                all_detections.append(kamus)
                if color_name == target_color: valid_targets.append(kamus)
            
            if not valid_targets:
                print(f"Skipping {target_color} - Not found")
                results.append(f"Skipped {target_color} (Not Found)")
                unavailable_items.append(target_item)
                continue

            target_dict = max(valid_targets, key=lambda x: x['y_pos'])
            target_index = all_detections.index(target_dict)
            draw_and_show_target(img_path, all_detections, target_index)

            # Move and Pick
            target_rad = [math.radians(x) for x in target_dict["prediction"][0]]
            servo.value = -0.14
            rtde_c.moveJ(target_rad, speed=0.6, acceleration=0.5)
            
            offset = knn.get_correction(target_dict["position"], target_dict["y_pos"])
            if any(o != 0 for o in offset):
                p = rtde_r.getActualTCPPose()
                p[0]+=offset[0]; p[1]+=offset[1]; p[2]+=offset[2]
                rtde_c.moveL(p, speed=0.2, acceleration=0.2)
            
            p = rtde_r.getActualTCPPose(); p_before = list(p)
            p[2] -= 0.065; rtde_c.moveL(p, speed=0.1, acceleration=0.1)
            servo.max(); sleep(0.8)
            p_after = rtde_r.getActualTCPPose()
            knn.log_error(target_dict["position"], target_dict["y_pos"], p_before, p_after)

            p[2] += 0.15; rtde_c.moveL(p, speed=0.3, acceleration=0.3)
            rtde_c.moveJ(target_drop_pos, speed=0.8, acceleration=0.8)
            servo.mid(); sleep(0.5)
            results.append(f"Picked {target_color}")

    except Exception as e:
        print(f"FATAL: {e}")
        return f"Error: {e}", unavailable_items
    finally:
        try: rtde_c.stopScript()
        except: pass
        cv2.destroyAllWindows()
        try: servo.close()
        except: pass

    return "\n".join(results), unavailable_items

# --- MAIN BLOCK FOR SUBPROCESS EXECUTION ---
if __name__ == "__main__":
    # This block allows the script to be run as: 
    # python3 robot_executor.py input.json output.json
    
    # --- Start of sys.argv Fix ---
    # Initialize file variables to None to handle index-out-of-range errors in the exception block
    input_file = None
    output_file = None
    
    try:
        # Check for minimum required arguments (script name + input file + output file)
        if len(sys.argv) < 3:
            # If arguments are missing, raise a clear error
            raise IndexError("Missing required arguments: input_file (sys.argv[1]) and output_file (sys.argv[2]).")

        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        # Load the Bill of Materials
        with open(input_file, 'r') as f:
            bom_list = json.load(f)
            
        # Execute the main robot logic
        log, unavailable = execute_flower_order(bom_list)
        
        # Write successful results to the output file
        result = {"log": log, "unavailable": unavailable}
        with open(output_file, 'w') as f:
            json.dump(result, f)
            
    except Exception as e:
        # If any error occurred, prepare the error response
        err_res = {"log": f"Critical Script Failure: {type(e).__name__}: {str(e)}", "unavailable": []}
        
        # Attempt to write the error response ONLY if output_file was successfully assigned
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(err_res, f)
            except Exception as write_err:
                # This inner except handles the case where we can't write to the file
                # due to permissions or a bad path, *after* output_file was set.
                print(f"Could not write error file to {output_file}: {write_err}")
        else:
             # If output_file was not set (i.e., sys.argv[2] failed), print the failure
             print(f"Critical Script Failure, no output file path available: {e}")
    # --- End of sys.argv Fix ---