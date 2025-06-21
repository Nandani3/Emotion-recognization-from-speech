
#  Emotion Recognition from Speech using CNN and MFCC

This project aims to detect human emotions from voice recordings using machine learning and deep learning techniques. By analyzing speech features like tone, pitch, and intensity, the model can classify audio samples into emotional states such as **Happy**, **Sad**, **Angry**, **Fear**, and more.

Emotion-aware systems are essential for building intelligent virtual assistants, mental health monitors, and empathetic AI interfaces.

---

##  Project Structure

emotion_recognition_speech/
├── data/ # Raw or downloaded speech data (e.g., RAVDESS)
├── features/ # MFCC extracted features stored as numpy arrays
├── models/ # Trained model weights or checkpoints
├── notebooks/ # Jupyter notebooks for EDA and model training
├── utils.py # Helper functions for preprocessing and feature extraction
├── train_model.py # Script for training CNN model
├── predict.py # Real-time prediction from mic input
├── requirements.txt # List of required Python packages
└── README.md # Project documentation


---

##  Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
Major Libraries Used:
Python 3.

1 - TensorFlow / Keras

2 - Librosa

3 -NumPy

4 - Pandas

5 - Scikit-learn

6 - Matplotlib / Seaborn

## Set up instructions ##
1 - Clone the Repository
git clone https://github.com/Nandani3/emotion-recognition-from-speech.git
cd emotion-recognition-from-speech


2 - Install Dependencies
pip install -r requirements.txt


3 - Download Dataset:

Download the RAVDESS dataset or use any other labeled speech dataset and place it inside the data/ directory.

4 - Extract Features:

Run the feature extraction script or use the preprocessing steps in notebooks to convert raw audio into MFCC features.

5 - Train the Model:
python train_model.py

6 - Predict Emotion:

You can run the prediction script using a pre-trained model:
python predict.py

## Usage
1 -Train your model using train_model.py

2 -Record your voice or input a .wav file

3 -Run predict.py to identify emotion

4 -Modify or extend the model for more complex speech emotion datasets


👥 Contributors
Nandani Bisht

📄 License
This project is licensed under the MIT License, which means you are free to use, modify, and distribute it for personal or commercial use.







