import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from keras.initializers import Orthogonal
import pydub
import io
import subprocess

# Update these paths to match your Mac directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, 'labels.csv')
AUDIO_DIR = os.path.join(BASE_DIR, 'audio')
MODEL_PATH = os.path.join(BASE_DIR, 'modddrnn-fin.h5')

# Load labels
labels = pd.read_csv(LABELS_PATH)
print(labels.head())
print(labels['label'].value_counts())

# Add these at the start of your processing code
def check_ffmpeg():
    """Check if ffmpeg is properly installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return True
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg using: brew install ffmpeg")
        return False

# Check ffmpeg before processing
if not check_ffmpeg():
    exit(1)

# Preprocess data
spectrograms_list = []
label_list = []
processed_files = 0
failed_files = 0

def convert_mp3_to_wav(mp3_path):
    """Convert MP3 to WAV format using pydub"""
    try:
        audio = pydub.AudioSegment.from_mp3(mp3_path)
        wav_data = io.BytesIO()
        audio.export(wav_data, format='wav')
        wav_data.seek(0)
        return wav_data
    except Exception as e:
        print(f"Error converting {mp3_path}: {e}")
        return None

def convert_all_mp3_to_wav():
    for filename in os.listdir(AUDIO_DIR):
        if filename.lower().endswith('.mp3'):
            mp3_path = os.path.join(AUDIO_DIR, filename)
            wav_path = os.path.join(AUDIO_DIR, filename[:-4] + '.wav')
            audio = pydub.AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format='wav')
            print(f"Converted {filename} to WAV")

print("Checking audio files:")
for index, row in labels.iterrows():
    filename = os.path.join(AUDIO_DIR, row['filename'])
    exists = os.path.exists(filename)
    print(f"{row['filename']}: {'Found' if exists else 'Not Found'}")

for index, row in labels.iterrows():
    filename = os.path.join(AUDIO_DIR, row['filename'])
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        failed_files += 1
        continue
        
    try:
        if filename.lower().endswith('.mp3'):
            wav_data = convert_mp3_to_wav(filename)
            if wav_data is None:
                failed_files += 1
                continue
            waveform, sample_rate = torchaudio.load(wav_data)
        else:
            waveform, sample_rate = torchaudio.load(filename, normalize=True)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Create mel spectrogram
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )(waveform)
        
        # Convert to decibels
        mel_spectrogram = transforms.AmplitudeToDB()(mel_spectrogram)
        
        # Pad or truncate
        if mel_spectrogram.shape[2] > fixed_len:
            mel_spectrogram = mel_spectrogram[:, :, :fixed_len]
        else:
            padding = fixed_len - mel_spectrogram.shape[2]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
        
        # Reshape for LSTM input (time_steps, features)
        mel_spectrogram = mel_spectrogram.squeeze(0).permute(1, 0)  # Shape: (1000, 128)
        
        spectrograms_list.append(mel_spectrogram.numpy())
        label_list.append(row['label'])
        processed_files += 1
        
    except Exception as e:
        print(f"Error processing file {filename}: {str(e)}")
        failed_files += 1
        continue

print(f"\nProcessing Summary:")
print(f"Successfully processed: {processed_files} files")
print(f"Failed to process: {failed_files} files")

# Only continue if we have processed files
if processed_files == 0:
    print("No files were successfully processed. Please check your audio files and try again.")
    exit(1)

# Convert to numpy arrays
spectrograms_array = np.array(spectrograms_list)
label_array = np.array(label_list)

print("\nFinal data shapes:")
print("Spectrograms shape:", spectrograms_array.shape)
print("Labels shape:", label_array.shape)

# Only proceed with model training if we have data
if len(spectrograms_array) == 0:
    print("No data to train on. Exiting.")
    exit(1)

label_encoder = LabelEncoder()
label_array_encoded = label_encoder.fit_transform(label_array)

X_train, X_test, y_train, y_test = train_test_split(spectrograms_array, label_array_encoded, test_size=0.2, stratify=label_array_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])
print("Original shapes:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)
num_features = X_train.shape[2]
print("Number of features in X_train:", num_features)

# Fix the reshaping
print("Before reshaping:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

# Correct reshaping for LSTM input (batch_size, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

print("\nAfter reshaping:")
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

def create_model():
    model = Sequential([
        # Input shape should be (timesteps, features)
        LSTM(128, return_sequences=True, kernel_initializer=Orthogonal(), 
             input_shape=(1000, 128)),  # Changed input shape
        Dropout(0.5),
        LSTM(64, return_sequences=True, kernel_initializer=Orthogonal()),
        Dropout(0.5),
        LSTM(32, kernel_initializer=Orthogonal()),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(5, activation='softmax')  # 5 classes
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train the model
model = create_model()
print("\nModel architecture:")
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

model.save(MODEL_PATH)

model_path = MODEL_PATH
model = load_model(model_path)

fixed_length=1000
def predict(audio_file):
    try:
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
        
        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Create mel spectrogram
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )(waveform)
        
        # Convert to decibels
        mel_spectrogram = transforms.AmplitudeToDB()(mel_spectrogram)
        
        # Pad or truncate to fixed length
        if mel_spectrogram.shape[2] > fixed_len:
            mel_spectrogram = mel_spectrogram[:, :, :fixed_len]
        else:
            padding = fixed_len - mel_spectrogram.shape[2]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
        
        # Reshape for model input (batch_size, time_steps, features)
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1)  # Reshape to (1, 1000, 128)
        mel_spectrogram = mel_spectrogram.numpy()
        
        # Make prediction
        prediction = model.predict(mel_spectrogram)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
        
        return predicted_class[0]
    except Exception as e:
        print(f"Error predicting for file: {audio_file}")
        print(e)
        return "Unknown"  # Return "Unknown" instead of None

filename = os.path.join(AUDIO_DIR, labels['filename'][0])
print(f"Testing file: {labels['filename'][0]}")
print(f"Prediction: {predict(filename)}")

true_labels_numeric = label_encoder.transform(labels['label'])
predictions = []

# Make predictions for all files
for idx in range(len(labels)):
    audio_file = os.path.join(AUDIO_DIR, labels['filename'][idx])
    pred = predict(audio_file)
    if pred != "Unknown":
        try:
            pred_numeric = label_encoder.transform([pred])[0]
            predictions.append(pred_numeric)
        except ValueError as e:
            print(f"Error with prediction for {labels['filename'][idx]}: {e}")
            predictions.append(label_encoder.transform(["Normal"])[0])  # Default to "Normal" class

# Calculate accuracy and confusion matrix
if predictions:
    accuracy = accuracy_score(true_labels_numeric[:len(predictions)], predictions)
    print(f"\nOverall Accuracy: {accuracy:.2f}")

    conf_matrix = confusion_matrix(true_labels_numeric[:len(predictions)], predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)

# Print detailed results
print("\nDetailed Predictions:")
for idx in range(len(labels)):
    print(f"File: {labels['filename'][idx]}")
    print(f"True Label: {labels['label'][idx]}")
    pred = predict(os.path.join(AUDIO_DIR, labels['filename'][idx]))
    print(f"Predicted: {pred}")
    print("---")
