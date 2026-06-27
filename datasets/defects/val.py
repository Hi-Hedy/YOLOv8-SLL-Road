"""Validation script for trained YOLOv8-SLL model."""
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(r"ultralytics-main-SLL\ultralytics\nn\extra_modules"))

from ultralytics import YOLO

def main():
    model = YOLO(r'weights\best.pt')
    print("Model classes:", model.names)

    model.val(
        data=r'defects.yaml',
        split='test',
        batch=32,
        imgsz=640,
        workers=0,
        device=0,
        project='runs/detect/val',
        name='SLL_test',
    )

if __name__ == '__main__':
    main()
