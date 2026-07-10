import cv2
from ultralytics import YOLO
import torch

print("Open cv", cv2.__version__)
print("Torch",torch.__version__)
print("Cuda disponivel",torch.cuda.is_available())

model = YOLO("yolov8n.pt")

print("Yolo carregado com sucesso")