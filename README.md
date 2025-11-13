# 🧠 Object Classification & Live Detection using AlexNet (PyTorch)

This project builds a **custom AlexNet-based deep learning model** to classify objects into three categories:

- **Water Bottle**
- **Mobile**
- **Nothing**

The system includes:

- A **complete training pipeline**
- **Dataset preprocessing & cleaning**
- **Model evaluation with a confusion matrix**
- **Real-time live object detection using webcam**  

---

## 📂 Dataset Overview

The dataset is preprocessed and stored in:

It contains:

- **X** → images  
- **Y** → labels (0, 1, 2)

During preprocessing:

- Images are converted to **grayscale**
- Resized to **128 × 128**
- Normalized using mean = 0.5 and std = 0.5

Images are split as:

- **85% → Training**
- **15% → Testing**

---

## 🧠 Model Architecture — Custom AlexNet Variant

The model is a simplified and optimized AlexNet-like CNN:

### 🔹 Convolution Layers
- Conv2d(1 → 64)  
- Conv2d(64 → 128)  
- Conv2d(128 → 256)  
- Conv2d(256 → 512)  
- ReLU + MaxPool used throughout  

### 🔹 Fully Connected Layers
- Linear → 4096  
- Linear → 4096  
- Linear → 3 (number of classes)  
- Dropout applied for regularization  

This architecture is designed to perform well on grayscale object recognition tasks.

---

## 🏋️ Training Details

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Epochs:** 5
- **Batch Size:** 64
- **Device:** MPS (Apple Silicon) or CPU

Each epoch prints:

- Training loss  
- Test accuracy  

The training loop collects predictions to generate the confusion matrix.

---

## 📊 Evaluation: Confusion Matrix

After training, predictions from the entire test set are compared to true labels.

A **normalized confusion matrix** is plotted:

- Helps understand per-class performance  
- Useful for understanding misclassification patterns  

---

## 🎥 Real-Time Live Detection

A webcam-based detection script is included.

### Steps:

1. Capture frames from webcam  
2. Preprocess using same transforms  
3. Forward pass through the trained model  
4. Display predicted label on screen  

Class labels used:

Press **Q** to exit live detection.

---

## 💾 Saving the Model

The trained model is saved as:

It is loaded automatically during live detection.

---

## 🛠️ Technologies Used

- **PyTorch**
- **Torchvision**
- **OpenCV**
- **NumPy**
- **Scikit-Learn**
- **Matplotlib**

---

## 🚀 Potential Improvements

- Add bounding-box based object detection  
- Switch to MobileNetV2 / EfficientNet for faster performance  
- Deploy as a desktop or web application (Streamlit/Gradio)  
- Add background subtraction for cleaner object segmentation  

---

If you want:

✅ A downloadable `README.md`  
✅ A GitHub-ready project structure  
✅ A combined pipeline for multiple models  

Just let me know!  
