"""Prediction script for trained YOLOv8-SLL model."""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.abspath(r"ultralytics-main-SLL\ultralytics\nn\extra_modules"))

from ultralytics import YOLO

# Load trained model
model = YOLO(r'weights\best.pt')

# Run prediction on test images
results = model.predict(
    source=r'\test\images',
    save=True,
    project='runs/detect/predict',
    name='SLL_predict',
)
