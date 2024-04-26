# In terminal: yolo predict model=yolov8n.pt source='apples_captures/testing.jpeg'

from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the input image
image_path = 'apples_captures/testing.jpeg'
image = cv2.imread(image_path)

# Perform object detection
results = model(image)
# results = model(source='apples_captures/testing.jpeg', show=True, conf=0.4, save=True, persist=True)