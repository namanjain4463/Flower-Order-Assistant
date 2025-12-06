import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import os
import sys

# Define the architecture used in robot_executor.py
# This is necessary to correctly load the state dictionary
class YOLOtoJointMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.Linear(32, 128), nn.Dropout(0.3),
                                 nn.ReLU(), nn.Linear(128, 16), nn.Linear(16, 6))
    def forward(self, x): 
        return self.net(x)

def resave_mlp_model(original_path, new_path):
    """
    Loads the MLP model and resaves its state_dict and scalers in a clean,
    highly compatible format to avoid C++ class instantiation errors.
    """
    print(f"--- Processing MLP Model: {original_path} ---")
    
    # 1. Load the file, mapping all tensors to the CPU
    try:
        # Load the checkpoint file
        ckpt = torch.load(original_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"ERROR: Could not load {original_path}. PyTorch error: {e}")
        print("Please ensure the file exists and is not corrupted.")
        return

    # 2. Extract state_dict and scalers
    if isinstance(ckpt, dict) and "model" in ckpt:
        # Preferred format: state_dict is under the 'model' key
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        # Fallback 1: Check if keys look like a state_dict directly
        # We need to filter to ensure we don't include scalers as part of the state_dict
        state_dict_keys = [k for k in ckpt.keys() if k.startswith("net.")]
        if state_dict_keys:
             state_dict = {k: ckpt[k] for k in state_dict_keys}
        else:
             print(f"ERROR: Could not find a valid state_dict structure in {original_path}.")
             return
    else:
        print(f"ERROR: {original_path} has an unexpected structure (not a dictionary).")
        return
        
    # Create a new dictionary for saving, ensuring all necessary scalers are kept
    new_ckpt = {"model": state_dict}
    
    # Check and transfer scalers
    scaler_keys = ["x_scaler_mean", "x_scaler_scale", "y_scaler_mean", "y_scaler_scale"]
    missing_scalers = [k for k in scaler_keys if k not in ckpt]
    if missing_scalers:
        print(f"WARNING: Missing scalers in original file: {missing_scalers}. Model may not function correctly.")
    
    for key in scaler_keys:
        if key in ckpt:
            # Crucial: Convert scalers (which might be numpy or list) to simple PyTorch tensors
            new_ckpt[key] = torch.tensor(ckpt[key], dtype=torch.float32)

    # 3. Save the clean checkpoint
    try:
        # Use _use_new_zipfile_serialization=False for maximum compatibility (legacy format)
        torch.save(new_ckpt, new_path, _use_new_zipfile_serialization=False)
        print(f"SUCCESS: Cleaned MLP model saved to {new_path}")
        print("The original file has NOT been overwritten.")
    except Exception as e:
        print(f"ERROR: Could not save the new model file: {e}")

def resave_yolo_model(original_path, new_path):
    """Loads and resaves the YOLO model file."""
    print(f"--- Processing YOLO Model: {original_path} ---")
    
    # YOLO models are often stored differently, but Ultralytics provides a way to export the engine.
    # The simplest fix is to just load it, which triggers a compatibility check/update if needed.
    try:
        # Load the YOLO model (this often handles compatibility issues internally)
        model = YOLO(original_path)
        
        # This function saves the current model state in a compatible format
        # Use the 'state_dict' only for simplicity and compatibility
        # We use the default torch.save here as Ultralytics often handles its own serialization
        torch.save(model.state_dict(), new_path) 
        
        print(f"SUCCESS: Cleaned YOLO model saved to {new_path}")
        print("The original file has NOT been overwritten.")

    except Exception as e:
        print(f"ERROR: Could not load/resave {original_path}. YOLO/PyTorch error: {e}")
        print("Please ensure the file exists and is valid.")

if __name__ == "__main__":
    # Check for existence of the files
    if not os.path.exists("flower_joint_model.pth") or not os.path.exists("best_yolo.pt"):
        # This is a mock error, as we don't have the files, but the logic is sound.
        print("INFO: Skipping model resave. Model files (flower_joint_model.pth or best_yolo.pt) not found.")
        print("If you had the files, the script would run the following steps:")
        print("1. Process MLP Model: flower_joint_model.pth -> flower_joint_model_CLEAN.pth")
        print("2. Process YOLO Model: best_yolo.pt -> best_yolo_CLEAN.pt")
        print("You must re-run this script locally with your model files to generate the clean versions.")
        
        # NOTE: If this were running where files existed, this would run:
        # resave_mlp_model("flower_joint_model.pth", "flower_joint_model_CLEAN.pth")
        # resave_yolo_model("best_yolo.pt", "best_yolo_CLEAN.pt")
        # sys.exit(1)
        
    else:
        # Process the MLP model
        resave_mlp_model(
            original_path="flower_joint_model.pth", 
            new_path="flower_joint_model_CLEAN.pth"
        )

        # Process the YOLO model
        resave_yolo_model(
            original_path="best_yolo.pt", 
            new_path="best_yolo_CLEAN.pt"
        )
        
        print("\n--- ACTION REQUIRED ---")
        print("1. DELETE the original files: 'flower_joint_model.pth' and 'best_yolo.pt'.")
        print("2. RENAME the new files:")
        print("   - Rename 'flower_joint_model_CLEAN.pth' to 'flower_joint_model.pth'")
        print("   - Rename 'best_yolo_CLEAN.pt' to 'best_yolo.pt'")
        print("3. Rerun your Streamlit application.")