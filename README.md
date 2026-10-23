
# 🎙️ Emotion Detection from Speech using Deep Learning

This project detects human emotions from audio speech using a deep learning model. It leverages spectrogram analysis and a hybrid CNN-RNN architecture (SEDNet) to classify emotions accurately from voice signals.

## 🚀 Overview

The goal is to automatically recognize emotions from speech audio samples. The system processes `.wav` files to extract Mel spectrograms, then passes them through a deep learning model that classifies the emotions into categories like *happy*, *sad*, *angry*, etc.

## 📁 Dataset

**Toronto Emotional Speech Set (TESS)**  
- Contains recordings of 7 emotions: **Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad**
- Each audio file is a `.wav` file
- Dataset structure:
  ```
  dataset/
  ├── Angry/
  ├── Disgust/
  ├── Fear/
  ├── Happy/
  ├── Neutral/
  ├── Pleasant_Surprise/
  └── Sad/
  ```

## 🧠 Model Architecture

**SEDNet (Spectrogram-based Emotion Detection Network)**  
- **CNN layers**: For spatial feature extraction from spectrograms  
- **RNN layers (LSTM/GRU)**: For capturing temporal features  
- **Fully Connected Layers**: For final emotion classification  
- **Activation**: Softmax

## 🔧 Technologies Used

- Python
- TensorFlow / Keras
- Librosa (for audio processing)
- NumPy, Matplotlib
- Scikit-learn (for evaluation metrics)
- OpenCV (optional, for visualization)

## 📊 Training & Results

- **Input**: Mel Spectrograms (128 Mel bands)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Accuracy Achieved**: ~93% on test data
- **Augmentations Used**:
  - Time Stretching
  - Pitch Shifting
  - Background Noise Addition

## 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-score

## 💡 Key Features

- End-to-end audio-based emotion recognition
- Robust against variations in speaker tone and speed
- Can be extended to real-time speech emotion recognition

## 🛠️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/emotion-detection-from-speech.git
cd emotion-detection-from-speech

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Evaluate
python evaluate.py
```

## 🧪 Demo

You can test the model by placing a `.wav` file in the `/test_audio` directory and running:

```bash
python predict.py --file test_audio/sample.wav
```

## 📌 Future Work

- Integrate with real-time speech input (microphone)
- Deploy as a Flask web app or Streamlit GUI
- Support for multilingual emotion detection

## 📚 References

- [TESS Dataset](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF)
- Papers on SEDNet and Speech Emotion Recognition
- Librosa and TensorFlow documentation

Last updated: 2026-10-23 21:50 UTC
