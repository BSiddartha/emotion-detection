# emotion-detection

A real-time **emotion detection system** built using **Convolutional Neural Networks (CNN)**, **TensorFlow/Keras**, and **OpenCV**.  
The model identifies 7 human emotions from facial expressions:

- 😡 Angry  
- 🤢 Disgust  
- 😨 Fear  
- 😀 Happy  
- 😢 Sad  
- 😲 Surprise  
- 😐 Neutral  

This project uses the **FER-2013** dataset to train a CNN model and deploys it for **live detection using a webcam**.

---

## 🚀 Features

- Real-time face detection using OpenCV  
- Emotion classification using a trained CNN model  
- Works on any webcam  
- 7 emotion category predictions  
- End-to-end ML pipeline: training → saving model → deployment  
- Lightweight and runs on CPU  

---

## 📂 Dataset

**Dataset:** FER-2013 (Facial Expression Recognition Dataset)  
Source: https://www.kaggle.com/datasets/msambare/fer2013  

- 35,887 images  
- 48x48 pixel grayscale images  
- 7 emotion classes  
- Train and test split included

Folder structure:

fer2013/
train/
test/

---

## 🧠 Model Architecture

The CNN consists of:

- Conv2D → MaxPooling
- Conv2D → MaxPooling
- Conv2D → MaxPooling
- Flatten
- Dense layers
- Dropout for regularization
- Softmax output layer (7 classes)

Optimizer: **Adam**  
Loss: **Categorical Crossentropy**

---

## 📁 Project Structure

emotion_detection/
│── emotion310_env/ # Virtual environment

│── fer2013/ # Dataset

│── train_model.py # Training code

│── detect_emotion.py # Real-time detection code

│── emotion_model.h5 # Saved trained model

│── README.md # Project documentation

---

## 🛠️ Installation

### 1️⃣ Clone the repository

git clone https://https://github.com/BSiddartha/emotion-detection
cd emotion-detection

### 2️⃣ Create virtual environment (Python 3.10 recommended)

py -3.10 -m venv emotion_env
emotion_env\Scripts\activate

### 3️⃣ Install dependencies

pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn

### 🏋️‍♂️ Training the Model

Run the training script:

python train_model.py

This will generate:

emotion_model.h5

### 🎥 Running Real-Time Emotion Detection

python detect_emotion.py

Press q to quit the webcam window.


### 🔧 Future Improvements

- Replace Haar Cascade with MTCNN for better face detection

- Use Transfer Learning (VGG16, ResNet50) for higher accuracy

- Build a Flask web interface for browser-based detection

- Deploy model using TensorFlow Lite

- Add multiple face detection and emotion tracking

### 📝 License

This project is open-source and available under the MIT License.

### 🙌 Acknowledgments
- FER-2013 Dataset

- TensorFlow Documentation

- OpenCV Library