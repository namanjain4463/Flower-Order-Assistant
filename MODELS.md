# Model Files Documentation

This document describes the machine learning models used in the Flower Order Assistant system.

## 📁 Model Files

### 1. Joint Angle Predictor (`flower_joint_model_CLEAN.pth`)

**Type**: PyTorch Neural Network Model  
**Size**: ~27 KB  
**Architecture**: YOLOtoJointMLP

#### Purpose
Predicts the 6 joint angles for the UR5 robotic arm based on the normalized Y-position of detected flowers in the camera frame.

#### Architecture Details
```
Input Layer:    1 neuron (normalized Y position)
Hidden Layer 1: 32 neurons
Hidden Layer 2: 128 neurons (with 30% Dropout)
Activation:     ReLU
Hidden Layer 3: 16 neurons
Output Layer:   6 neurons (joint angles in radians)
```

#### Model Components
The `.pth` file contains:
- **model**: State dictionary with trained weights
- **x_scaler_mean**: Input feature mean for normalization
- **x_scaler_scale**: Input feature scale for normalization
- **y_scaler_mean**: Output feature mean for denormalization
- **y_scaler_scale**: Output feature scale for denormalization

#### Input
- Normalized Y-coordinate from YOLO bounding box (0-1 range)

#### Output
- Array of 6 joint angles in radians: `[j1, j2, j3, j4, j5, j6]`

#### Usage
```python
from robot_executor import load_mlp, predict_joints

# Load model and scalers
mlp, x_mean, x_scale, y_mean, y_scale = load_mlp("flower_joint_model_CLEAN.pth")

# Predict joint angles
y_normalized = 0.65  # Example Y position
joints = predict_joints([[y_normalized]], mlp, x_mean, x_scale, y_mean, y_scale)
print(joints)  # Output: [6 joint angles]
```

---

### 2. YOLO Flower Detector (`best_yolo_CLEAN.pt`)

**Type**: YOLO Object Detection Model  
**Size**: Not included in repository  
**Framework**: Ultralytics YOLO

#### Purpose
Detects flowers in camera images and classifies them by color (red, orange, pink, purple, white).

#### Model Details
- **Task**: Object Detection + Classification
- **Input**: RGB image (1280x720)
- **Output**: Bounding boxes with class labels and confidence scores

#### Output Format
For each detection:
- **xywh**: Normalized bounding box coordinates (center_x, center_y, width, height)
- **class**: Color label (red, orange, pink, purple, white)
- **confidence**: Detection confidence score

#### Usage
```python
from ultralytics import YOLO

model = YOLO("best_yolo_CLEAN.pt")
results = model.predict("image.jpg", conf=0.25)

for box in results[0].boxes:
    xn, yn, wn, hn = box.xywhn[0].tolist()
    color = results[0].names[int(box.cls)]
    conf = box.conf
    print(f"Detected {color} flower at ({xn}, {yn}) with confidence {conf}")
```

---

## 🔄 Model Training

### Joint Predictor Training

The MLP model was trained to map flower positions in the image to robot joint configurations:

1. **Data Collection**: Manual positioning of robot to known flower locations
2. **Feature Extraction**: Record Y-coordinates from camera frame
3. **Label Collection**: Record corresponding joint angles from robot
4. **Normalization**: Apply StandardScaler to inputs and outputs
5. **Training**: Train MLP with MSE loss and Adam optimizer
6. **Validation**: Test on held-out flower positions

### YOLO Training

The YOLO model was trained on a custom dataset:

1. **Dataset**: Images of flowers in various positions
2. **Annotations**: Bounding boxes with color labels
3. **Training**: Fine-tuned from YOLOv8 base model
4. **Validation**: Tested on separate validation set
5. **Export**: Exported to `.pt` format for inference

---

## 📊 Model Performance

### Joint Predictor
- **Accuracy**: High precision for vertical positioning
- **Error Compensation**: KNN system further improves accuracy over time
- **Inference Time**: <5ms per prediction

### YOLO Detector
- **Confidence Threshold**: 0.25 (configurable)
- **Classes**: 5 flower colors
- **Inference Time**: ~50-100ms per image on CPU

---

## 🔧 Model Management with Git LFS

Large model files are tracked using Git Large File Storage (Git LFS):

```bash
# Track model files
git lfs track "*.pth"
git lfs track "*.pt"

# Add and commit
git add .gitattributes
git add flower_joint_model_CLEAN.pth
git commit -m "Add model files with Git LFS"
git push
```

---

## 📝 Notes

- **Model Files**: The `.pth` file is included in the repository via Git LFS
- **YOLO Model**: The `.pt` file may need to be provided separately or trained
- **Updates**: Models can be retrained with new data to improve accuracy
- **Compatibility**: Ensure PyTorch and Ultralytics versions match training environment

---

## 🔮 Future Improvements

1. **Multi-object Tracking**: Track flowers across frames for smoother operation
2. **Depth Estimation**: Add Z-axis prediction for 3D positioning
3. **Model Quantization**: Optimize for faster inference on edge devices
4. **Online Learning**: Update models based on KNN error logs
5. **Ensemble Methods**: Combine multiple models for improved robustness

---

**Last Updated**: December 7, 2025
