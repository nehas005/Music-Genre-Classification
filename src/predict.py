import os
import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cnn_model.keras"
)

CLASSES_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "cnn_classes.npy"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: CNN model not found!")
    print(MODEL_PATH)
    exit()

if not os.path.exists(CLASSES_PATH):
    print("ERROR: CNN classes file not found!")
    print(CLASSES_PATH)
    exit()


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING CNN MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

print("Model loaded successfully!")
print("Number of classes:", len(classes))

print()
print("Genres:")

for i, genre in enumerate(classes):
    print(f"{i}: {genre}")


# ============================================================
# EXTRACT MEL-SPECTROGRAM
# ============================================================

def extract_mel_spectrogram(file_path):

    print()
    print("Loading audio file...")

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    print("Audio loaded successfully!")
    print("Sample rate:", sr)
    print("Audio duration:", round(len(audio) / sr, 2), "seconds")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# ============================================================
# PREPARE SPECTROGRAM
# ============================================================

def prepare_spectrogram(mel):

    # Normalize between 0 and 1

    mel_min = mel.min()
    mel_max = mel.max()

    if mel_max != mel_min:

        mel = (
            (mel - mel_min)
            /
            (mel_max - mel_min)
        )

    else:

        mel = np.zeros_like(mel)

    # Add channel dimension

    mel = mel[..., np.newaxis]

    # Resize to CNN input size

    mel = tf.image.resize(
        mel,
        [IMAGE_HEIGHT, IMAGE_WIDTH]
    ).numpy()

    return mel.astype(np.float32)


# ============================================================
# PREDICT GENRE
# ============================================================

def predict_genre(file_path):

    print()
    print("=" * 60)
    print("MUSIC GENRE PREDICTION")
    print("=" * 60)

    # Extract spectrogram

    mel = extract_mel_spectrogram(
        file_path
    )

    # Prepare CNN input

    spectrogram = prepare_spectrogram(
        mel
    )

    # Add batch dimension

    spectrogram = np.expand_dims(
        spectrogram,
        axis=0
    )

    print()
    print("Spectrogram shape:")
    print(spectrogram.shape)

    # Make prediction

    predictions = model.predict(
        spectrogram,
        verbose=0
    )

    # Get predicted class

    predicted_index = np.argmax(
        predictions[0]
    )

    predicted_genre = classes[
        predicted_index
    ]

    confidence = (
        predictions[0][predicted_index]
        * 100
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print()
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print()
    print(
        "Predicted Genre:",
        predicted_genre
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print()

    # Show all genre probabilities

    print("Genre probabilities:")
    print("-" * 40)

    sorted_indices = np.argsort(
        predictions[0]
    )[::-1]

    for index in sorted_indices:

        probability = (
            predictions[0][index]
            * 100
        )

        print(
            f"{classes[index]:10s} : "
            f"{probability:.2f}%"
        )

    print()
    print("=" * 60)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MUSIC GENRE CLASSIFICATION")
    print("CNN Prediction System")
    print("=" * 60)

    print()
    print("Enter the path of an audio file.")
    print("Example:")
    print(
        r"C:\Users\neera\Downloads\song.wav"
    )

    print()

    audio_path = input(
        "Audio file path: "
    ).strip().strip('"')

    # Check file

    if not os.path.exists(audio_path):

        print()
        print("ERROR: Audio file not found!")
        print(audio_path)

        exit()

    # Check extension

    valid_extensions = (
        ".wav",
        ".mp3",
        ".au"
    )

    if not audio_path.lower().endswith(
        valid_extensions
    ):

        print()
        print(
            "ERROR: Unsupported audio format!"
        )

        print(
            "Use .wav, .mp3 or .au"
        )

        exit()

    # Predict

    try:

        predict_genre(
            audio_path
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("ERROR DURING PREDICTION")
        print("=" * 60)

        print(e)