"""Training script for YOLOv8-SLL GPR defect detection.
Usage:
    python train_defects.py
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random, numpy as np, torch  # ← 新增

# ====== 随机种子设置 ======
def set_seed(s):
    random.seed(s);
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = None  # 不设种子
# SEED = 42      # 切换：注释上面，取消这行
# SEED = 1234
# SEED = 2025
# ==========================

if SEED is not None:
    set_seed(SEED)
# ====== 种子设置结束 ======

from ultralytics import YOLO

if __name__ == '__main__':
    MODEL_CFG = r'yolov8-SLL.yaml'

    model = YOLO(MODEL_CFG)
    model.load('yolov8n.pt')

    model.train(
        data=r'defects.yaml',
        imgsz=640,
        epochs=200,
        batch=32,
        workers=0,
        device='0',
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.0005,
        lr0=0.01,
        cos_lr=True,
        amp=False,
        close_mosaic=10,
        label_smoothing=0.1,
        project='runs/detect/train1',
    )



#
# from ultralytics import YOLO
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# def train():
#     # Load a model
#     # model = YOLO('yolov8n.yaml')  # build a new model from YAML
#     # model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
#     model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
#
#     # Train the model0
#     results = model.train(data=r'D:\Anaconda3\YOLOv8\ultralytics-main\datasets\defects\defects.yaml',
#                           pretrained=r'D:\Anaconda3\YOLOv8\ultralytics-main\datasets\defects\yolov8n.pt',
#                           epochs=300, imgsz=640, device=0, batch=16, workers=0, amp=False)
#
# if __name__ == "__main__":
#     train()