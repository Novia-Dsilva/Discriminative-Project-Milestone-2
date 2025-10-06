# Discriminative-Project-Milestone-2


## Objective

To develop an automated celebrity detection system capable of identifying multiple individuals within a single image. The project involves creating a synthetic multi-person dataset through image concatenation, applying data augmentation techniques to enhance model generalization, custom training a YOLOv8 object detection network on 46 celebrity classes, and deploying the trained model to accurately predict both the identities and spatial locations of all celebrities present in test images.

## Project Overview

This project implements an end-to-end pipeline for multi-celebrity detection and identification using deep learning.

## Methodology

1. **Dataset Creation**: Concatenated individual celebrity images into 2×2 grid composites, generating 800 training, 200 validation, and 200 test images with 4 celebrities per image.

2. **Data Augmentation**: Applied on-the-fly augmentation during training including rotation (5°), translation (10%), scaling, horizontal flipping, color transformations (HSV), mosaic, and mixup augmentation.

3. **Model Training**: Custom trained YOLOv8 nano model for 50 epochs on 46 celebrity classes using transfer learning from pretrained COCO weights.

4. **Inference**: Deployed model to detect and identify all celebrities in test images, outputting celebrity IDs, names, confidence scores, and bounding box coordinates.

## Results

- **mAP50**: 83.91%
- **Precision**: 77.73%
- **Recall**: 76.88%

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- PIL

## Usage
```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model.predict("path/to/image.jpg")
