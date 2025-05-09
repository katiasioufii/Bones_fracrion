# # 🦴 Bone Fracture Detection using Image Processing & Deep Learning

This project focuses on detecting **bone fractures** in X-ray images. It combines classical image preprocessing with deep learning models like **YOLOv8** and **CNNs** to identify potential fracture regions.

---

## 🔍 Project Pipeline

### 🧪 1. Image Preprocessing

We first enhance the X-ray images using contrast and edge detection techniques to highlight important features.

```python
import cv2
import numpy as np

def preprocess_image(img):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Apply Gaussian Blur to reduce noise
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Detect edges using Canny
    edges = cv2.Canny(img, 100, 200)

    # Combine original image with edges
    img = cv2.addWeighted(img, 0.8, edges, 0.2, 0)

    # Normalize and convert to uint8
    img = img / 255.0
    return (img * 255).astype(np.uint8)
