### 🖐️ Hand Gesture Recognition using Deep Learning

A high-performance Computer Vision project that classifies hand gestures in real-time using a custom Convolutional Neural Network (CNN). This model achieves 99.6% accuracy on the LeapGestRecog dataset and is optimized for low-latency live inference.

### 📌 Project Overview

The goal of this project is to bridge the gap between human intent and machine action through touchless interfaces. By processing infrared hand imagery, the system can distinguish between 10 distinct gestures (Palm, Fist, OK, etc.) with near-perfect precision.

### Key Technical Achievements:

99.63% Validation Accuracy achieved within 5 epochs.

Real-time Inference integrated via OpenCV webcam feed.

Robust Preprocessing pipeline handling 40,000+ images.

### 🛠️ Tech StackLanguage: 
Python 3.xDeep Learning: TensorFlow / Keras

Computer Vision: OpenCV

Data Science: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

### 📂 Dataset Pipeline

The project utilizes the LeapGestRecog dataset. The pipeline includes:

1. Automated Extraction: Scripts to unzip and synchronize image paths with directory-based labels.

2. Image Standardization: Grayscale conversion and resizing to $64 \times 64$ pixels.
 
3. Normalization: Scaling pixel values to the range $[0, 1]$ to stabilize gradient descent
 
4. One-Hot Encoding: Converting categorical labels into a format suitable for categorical cross-entropy loss.

### 🧠 Model Architecture

The model uses a "Deep Feature Extractor" approach:

3 Convolutional Layers: Increasing filter depth (32, 64, 128) to capture hierarchical spatial features.

Batch Normalization: Applied after every Conv layer to prevent internal covariate shift and accelerate training.

Dropout (0.5): Strategic neurons are disabled during training to force the model to learn redundant, robust features, preventing overfitting.

Softmax Output: A 10-way dense layer providing probability distributions for each gesture.

📊 Evaluation

Confusion Matrix

The model demonstrates exceptional class separation. 

Out of a 4,000-image test set, errors were restricted almost exclusively to minor confusion between "Palm" and "Index" gestures.

Performance Metrics

Precision: 0.99

Recall: 0.99

F1-Score: 0.99

🚀 How to Run

Clone the Repository:

Bash

git clonehttps://github.com/yourusername/hand-gesture-recognition.git

Install Dependencies:

Bash

pip install -r requirements.txt

Run Live Test:

Bash

python live_inference.py

### 🔮 Future Work

Mobile Deployment: Converting the model to .tflite for Android/iOS usage.

Gesture-to-Action: Mapping specific gestures to OS commands (e.g., volume control, slide transitions).

Complex Backgrounds: Implementing Data Augmentation to improve performance in non-infrared, "messy" environments.

### Author

Muhammad Abdulkareem 
