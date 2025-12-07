# Model Files

This directory contains the machine learning models required for the Flower Order Assistant.

## Required Models

### 1. MLP Joint Predictor (`flower_joint_model_CLEAN.pth`)

**Purpose**: Predicts UR5 robot joint angles from YOLO detection coordinates

**Details**:
- **Architecture**: YOLOtoJointMLP (1 → 32 → 128 → 16 → 6)
- **Input**: Normalized Y position from YOLO detection (1D tensor)
- **Output**: 6 joint angles in radians
- **Includes**: StandardScaler parameters (x_scaler_mean, x_scaler_scale, y_scaler_mean, y_scaler_scale)
- **File Size**: ~50 KB
- **Status**: ✅ Included in repository

**Loading**:
```python
from robot_executor import load_mlp
mlp, x_mean, x_scale, y_mean, y_scale = load_mlp("flower_joint_model_CLEAN.pth")
```

### 2. YOLO Object Detection (`best_yolo_CLEAN.pt`)

**Purpose**: Detects and classifies flowers by color in camera images

**Details**:
- **Framework**: Ultralytics YOLOv8
- **Classes**: red, orange, pink, purple, white
- **Input**: 1280x720 RGB images
- **Output**: Bounding boxes (xywh normalized), class labels, confidence scores
- **File Size**: ~6 MB (typical for YOLOv8n)
- **Status**: ⚠️ Not included (large file - needs to be obtained separately)

**Training**:
If you need to train your own model:
```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')

# Train on your flower dataset
results = model.train(
    data='flowers.yaml',  # Dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    name='flower_detector'
)

# Export
model.export(format='pt')
```

**Dataset Structure** (flowers.yaml):
```yaml
path: ./flowers_dataset
train: images/train
val: images/val

names:
  0: red
  1: orange
  2: pink
  3: purple
  4: white
```

## Obtaining the YOLO Model

### Option 1: Download Pre-trained Model
If a pre-trained model is available from the project maintainer, download it and place it in the project root:
```bash
# Place the downloaded file here:
# Flower-Order-Assistant/best_yolo_CLEAN.pt
```

### Option 2: Train Your Own
1. Collect flower images for each color (100+ images per class recommended)
2. Annotate bounding boxes using tools like [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/heartexlabs/labelImg)
3. Export in YOLO format
4. Train using the script above
5. Save the best model as `best_yolo_CLEAN.pt`

### Option 3: Use a Placeholder for Testing
For testing the pipeline without the robot:
```python
# Create a dummy YOLO model (won't actually detect flowers)
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Download base model
model.save('best_yolo_CLEAN.pt')
```

## Model File Locations

```
Flower-Order-Assistant/
├── flower_joint_model_CLEAN.pth  ✅ (included)
└── best_yolo_CLEAN.pt            ⚠️ (download/train required)
```

## Verification

After obtaining the models, verify they load correctly:

```python
# Test MLP model
from robot_executor import load_mlp
try:
    mlp, xm, xs, ym, ys = load_mlp("flower_joint_model_CLEAN.pth")
    print("✅ MLP model loaded successfully")
except Exception as e:
    print(f"❌ MLP model error: {e}")

# Test YOLO model
from ultralytics import YOLO
try:
    yolo = YOLO("best_yolo_CLEAN.pt")
    print("✅ YOLO model loaded successfully")
    print(f"   Classes: {yolo.names}")
except Exception as e:
    print(f"❌ YOLO model error: {e}")
```

## Model Performance

### Expected Accuracy
- **YOLO Detection**: >90% mAP@0.5 on validation set
- **Joint Prediction**: <5° average error on calibrated positions
- **KNN Compensation**: Improves accuracy by 2-3mm after 20+ picks

### Troubleshooting

**Issue**: Model file not found
- **Solution**: Ensure files are in project root, check file names

**Issue**: Model loading error
- **Solution**: Verify PyTorch version compatibility, check file isn't corrupted

**Issue**: YOLO detects wrong colors
- **Solution**: Retrain with more diverse dataset, adjust confidence threshold

**Issue**: Joint predictions cause collisions
- **Solution**: Recalibrate camera position, retrain MLP with new data

## Contributing Models

If you've trained improved models:
1. Document the training dataset and parameters
2. Share model performance metrics
3. Consider uploading to a model hosting service (HuggingFace, etc.)
4. Update this README with download links

---

**Note**: Due to GitHub file size limits, large models may need to be hosted externally (Git LFS, HuggingFace, Google Drive, etc.).
