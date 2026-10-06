import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# Genre classes
# ============================================================

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 22050

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

RANDOM_STATE = 42


# ============================================================
# Extract Mel-Spectrogram
# ============================================================

def extract_mel_spectrogram(file_path):

    """
    Load an audio file and convert it into
    a log-scaled Mel-spectrogram.
    """

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

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
# Prepare Spectrogram for CNN
# ============================================================

def prepare_spectrogram(mel):

    """
    Normalize the Mel-spectrogram and resize it
    to 128 x 128 pixels.
    """

    # --------------------------------------------------------
    # Normalize between 0 and 1
    # --------------------------------------------------------

    mel_min = mel.min()
    mel_max = mel.max()

    if mel_max != mel_min:

        mel = (
            mel - mel_min
        ) / (
            mel_max - mel_min
        )

    else:

        mel = np.zeros_like(mel)


    # --------------------------------------------------------
    # Add channel dimension
    # Shape becomes:
    # (height, width, 1)
    # --------------------------------------------------------

    mel = mel[..., np.newaxis]


    # --------------------------------------------------------
    # Resize to 128 x 128
    # --------------------------------------------------------

    mel = tf.image.resize(
        mel,
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ]
    ).numpy()


    return mel.astype(np.float32)


# ============================================================
# Load GTZAN Dataset
# ============================================================

def load_dataset(dataset_path):

    """
    Load all GTZAN audio files.

    Expected structure:

    dataset/
        GTZAN/
            Data/
                genres_original/
                    blues/
                    classical/
                    country/
                    disco/
                    hiphop/
                    jazz/
                    metal/
                    pop/
                    reggae/
                    rock/
    """

    X = []
    y = []

    print("=" * 60)
    print("Loading GTZAN dataset")
    print("=" * 60)

    print()
    print("Dataset path:")
    print(dataset_path)

    print()


    # --------------------------------------------------------
    # Loop through all genres
    # --------------------------------------------------------

    for label, genre in enumerate(GENRES):

        genre_path = os.path.join(
            dataset_path,
            genre
        )


        # ----------------------------------------------------
        # Check whether genre folder exists
        # ----------------------------------------------------

        if not os.path.isdir(genre_path):

            print(
                f"WARNING: Folder not found: {genre_path}"
            )

            continue


        # ----------------------------------------------------
        # Find audio files
        # ----------------------------------------------------

        audio_files = [

            f
            for f in os.listdir(genre_path)

            if f.lower().endswith(
                (
                    ".wav",
                    ".au",
                    ".mp3"
                )
            )

        ]


        print(
            f"{genre}: {len(audio_files)} files"
        )


        # ----------------------------------------------------
        # Process each audio file
        # ----------------------------------------------------

        for filename in audio_files:

            file_path = os.path.join(
                genre_path,
                filename
            )

            try:

                # Extract Mel-spectrogram
                mel = extract_mel_spectrogram(
                    file_path
                )


                # Convert to CNN input
                spectrogram = prepare_spectrogram(
                    mel
                )


                # Add data
                X.append(
                    spectrogram
                )

                y.append(
                    label
                )


            except Exception as e:

                print(
                    f"Error processing {file_path}: {e}"
                )


    # --------------------------------------------------------
    # Convert lists to NumPy arrays
    # --------------------------------------------------------

    X = np.array(X)
    y = np.array(y)


    return X, y


# ============================================================
# Build CNN Model
# ============================================================

def build_cnn_model():

    """
    Create the Convolutional Neural Network.
    """

    model = tf.keras.Sequential([

        # ----------------------------------------------------
        # First convolution block
        # ----------------------------------------------------

        tf.keras.layers.Input(
            shape=(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                1
            )
        ),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),


        # ----------------------------------------------------
        # Second convolution block
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),


        # ----------------------------------------------------
        # Third convolution block
        # ----------------------------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),


        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        tf.keras.layers.Flatten(),


        # ----------------------------------------------------
        # Fully connected layer
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),


        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        tf.keras.layers.Dropout(
            0.5
        ),


        # ----------------------------------------------------
        # Output layer
        # 10 genres
        # ----------------------------------------------------

        tf.keras.layers.Dense(
            len(GENRES),
            activation="softmax"
        )

    ])


    # --------------------------------------------------------
    # Compile model
    # --------------------------------------------------------

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )


    return model


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # Project root
    # ========================================================

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )


    # ========================================================
    # IMPORTANT:
    #
    # Your actual dataset is:
    #
    # dataset/GTZAN/Data/genres_original
    # ========================================================

    dataset_path = os.path.join(
        project_root,
        "dataset",
        "GTZAN",
        "Data",
        "genres_original"
    )


    # ========================================================
    # Models directory
    # ========================================================

    models_path = os.path.join(
        project_root,
        "models"
    )


    # ========================================================
    # Results directory
    # ========================================================

    results_path = os.path.join(
        project_root,
        "results"
    )


    # Create directories if necessary

    os.makedirs(
        models_path,
        exist_ok=True
    )

    os.makedirs(
        results_path,
        exist_ok=True
    )


    # ========================================================
    # Check dataset path
    # ========================================================

    print()
    print("=" * 60)
    print("CHECKING DATASET PATH")
    print("=" * 60)

    print(
        dataset_path
    )


    if not os.path.isdir(dataset_path):

        print()
        print("ERROR: Dataset folder was not found.")
        print()
        print(
            "Expected:"
        )
        print(
            "dataset/GTZAN/Data/genres_original"
        )

        raise FileNotFoundError(
            dataset_path
        )


    print()
    print("Dataset folder found!")


    # ========================================================
    # Load dataset
    # ========================================================

    print()
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)


    X, y = load_dataset(
        dataset_path
    )


    # ========================================================
    # Display dataset information
    # ========================================================

    print()
    print("=" * 60)
    print("DATASET LOADED")
    print("=" * 60)

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )


    # ========================================================
    # Make sure data was loaded
    # ========================================================

    if len(X) == 0:

        raise ValueError(
            "No audio files were loaded. "
            "Check the GTZAN dataset path."
        )


    # ========================================================
    # Train / Validation / Test split
    #
    # 70% Training
    # 15% Validation
    # 15% Testing
    # ========================================================

    print()
    print("=" * 60)
    print("SPLITTING DATASET")
    print("=" * 60)


    X_train, X_temp, y_train, y_temp = train_test_split(

        X,
        y,

        test_size=0.30,

        random_state=RANDOM_STATE,

        stratify=y

    )


    X_val, X_test, y_val, y_test = train_test_split(

        X_temp,
        y_temp,

        test_size=0.50,

        random_state=RANDOM_STATE,

        stratify=y_temp

    )


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # ========================================================
    # Build CNN
    # ========================================================

    print()
    print("=" * 60)
    print("BUILDING CNN")
    print("=" * 60)


    model = build_cnn_model()


    # Display architecture

    model.summary()


    # ========================================================
    # Early stopping
    # ========================================================

    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True

    )


    # ========================================================
    # Train CNN
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING CNN")
    print("=" * 60)


    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=25,

        batch_size=32,

        callbacks=[
            early_stopping
        ],

        verbose=1

    )


    # ========================================================
    # Evaluate CNN
    # ========================================================

    print()
    print("=" * 60)
    print("EVALUATING CNN")
    print("=" * 60)


    test_loss, test_accuracy = model.evaluate(

        X_test,

        y_test,

        verbose=0

    )


    print()
    print(
        f"CNN Test Accuracy: {test_accuracy:.4f}"
    )


    # ========================================================
    # Predictions
    # ========================================================

    print()
    print("Generating predictions...")


    predictions = model.predict(

        X_test,

        verbose=0

    )


    predicted_labels = np.argmax(

        predictions,

        axis=1

    )


    accuracy = accuracy_score(

        y_test,

        predicted_labels

    )


    print(
        f"CNN Accuracy: {accuracy:.4f}"
    )


    # ========================================================
    # Save CNN model
    # ========================================================

    model_file = os.path.join(

        models_path,

        "cnn_model.keras"

    )


    model.save(

        model_file

    )


    print()
    print(
        "CNN model saved:"
    )

    print(
        model_file
    )


    # ========================================================
    # Save class names
    # ========================================================

    classes_file = os.path.join(

        models_path,

        "cnn_classes.npy"

    )


    np.save(

        classes_file,

        np.array(GENRES)

    )


    print()
    print(
        "CNN classes saved:"
    )

    print(
        classes_file
    )


    # ========================================================
    # Save test data
    # ========================================================

    np.save(

        os.path.join(
            results_path,
            "cnn_X_test.npy"
        ),

        X_test

    )


    np.save(

        os.path.join(
            results_path,
            "cnn_y_test.npy"
        ),

        y_test

    )


    # ========================================================
    # Final message
    # ========================================================

    print()
    print("=" * 60)
    print("CNN TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Final Test Accuracy: {test_accuracy:.4f}"
    )

    print()
    print("Files created:")

    print(
        "models/cnn_model.keras"
    )

    print(
        "models/cnn_classes.npy"
    )

    print(
        "results/cnn_X_test.npy"
    )

    print(
        "results/cnn_y_test.npy"
    )

    print()
    print("=" * 60)