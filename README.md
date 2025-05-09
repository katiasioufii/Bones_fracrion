# 🦴 Bone Fracture Detection using Image Processing & Deep Learning

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
```

---

### 🗂️ 2. Dataset

You can download the dataset directly from [Kaggle](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project) using `kagglehub`:

```python
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("pkdarabi/bone-fracture-detection-computer-vision-project")

print("Path to dataset files:", path)
```

Make sure you have [Kaggle API credentials](https://www.kaggle.com/docs/api) set up before running the command.

---

### 🤖 3. YOLOv8 Detection

We use **YOLOv8** for object detection on processed images:

- Annotated datasets are used to train the model.
- Model detects and localizes potential fractures using bounding boxes.
- Easy to integrate and deploy.

> 🚧 **Training in progress...**  
> Results and model weights will be added soon.

---

### 🧠 4. CNN and Other Models

We're also experimenting with:

- CNN-based binary classifiers
- Transfer learning (e.g., ResNet, MobileNet)
- Ensemble models for improved performance

> 📌 **Updates coming soon after evaluation.**

---

## 📁 Project Structure

```
├── preprocessing/
│   └── preprocess.py
├── models/
│   ├── yolo_inference.py
│   └── cnn_model.py
├── data/
│   └── xray_images/
├── notebooks/
│   └── training_experiments.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bone-fracture-detection.git
cd bone-fracture-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

Follow the instructions above using `kagglehub`.

### 4. Preprocess images

```python
from preprocessing.preprocess import preprocess_image
```

### 5. Train models

- Use YOLOv8 via Ultralytics ([docs](https://docs.ultralytics.com/))
- Or run `cnn_model.py` to train the CNN

---

## 📊 Results

> Results will be documented here after training is complete. Check back soon!

---

## 🔮 Future Work

- Improve model accuracy with more diverse datasets
- Visualize model attention using Grad-CAM
- Optimize for mobile deployment (ONNX / TFLite)
- Train on more X-ray datasets from Kaggle or open medical repositories

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and open a pull request. For major changes, open an issue first to discuss what you’d like to change.

---
