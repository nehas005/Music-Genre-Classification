import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


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
# Extract Mel-Spectrogram
# ============================================================

def extract_mel_spectrogram(
    file_path,
    sample_rate=22050,
    n_mels=128,
    n_fft=2048,
    hop_length=512
):

    audio, sr = librosa.load(
        file_path,
        sr=sample_rate
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# ============================================================
# Convert Spectrogram to Features
# ============================================================

def spectrogram_to_features(mel_spectrogram):

    mean_features = np.mean(
        mel_spectrogram,
        axis=1
    )

    std_features = np.std(
        mel_spectrogram,
        axis=1
    )

    features = np.concatenate(
        [
            mean_features,
            std_features
        ]
    )

    return features


# ============================================================
# Build Feature Dataset
# ============================================================

def build_feature_dataset(dataset_path):

    X = []
    y = []

    print()
    print("Dataset path:")
    print(dataset_path)
    print()

    for genre in GENRES:

        genre_path = os.path.join(
            dataset_path,
            genre
        )

        if not os.path.isdir(genre_path):

            print(
                f"WARNING: Missing folder: {genre_path}"
            )

            continue

        audio_files = [
            f
            for f in os.listdir(genre_path)
            if f.lower().endswith(
                (".wav", ".au", ".mp3")
            )
        ]

        print(
            f"{genre}: {len(audio_files)} files"
        )

        for filename in audio_files:

            file_path = os.path.join(
                genre_path,
                filename
            )

            try:

                mel = extract_mel_spectrogram(
                    file_path
                )

                features = spectrogram_to_features(
                    mel
                )

                X.append(features)
                y.append(genre)

            except Exception as e:

                print(
                    f"Error processing {file_path}: {e}"
                )

    X = np.array(X)
    y = np.array(y)

    return X, y


# ============================================================
# Save Sample Spectrogram
# ============================================================

def save_sample_spectrogram(
    audio_file,
    output_file
):

    mel = extract_mel_spectrogram(
        audio_file
    )

    plt.figure(
        figsize=(10, 4)
    )

    librosa.display.specshow(
        mel,
        sr=22050,
        hop_length=512,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        "Mel-Spectrogram"
    )

    plt.tight_layout()

    plt.savefig(
        output_file
    )

    plt.close()


# ============================================================
# Main Program
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    # --------------------------------------------------------
    # CORRECT GTZAN DATASET PATH
    # --------------------------------------------------------

    dataset_path = os.path.join(
        project_root,
        "dataset",
        "GTZAN",
        "Data",
        "genres_original"
    )

    # --------------------------------------------------------
    # Results directory
    # --------------------------------------------------------

    results_path = os.path.join(
        project_root,
        "results"
    )

    os.makedirs(
        results_path,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    print("=" * 60)
    print("CHECKING DATASET PATH")
    print("=" * 60)

    print(dataset_path)

    if not os.path.isdir(dataset_path):

        print()
        print("ERROR: Dataset folder not found!")
        print(dataset_path)
        exit()

    print()
    print("Dataset folder found!")

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BUILDING GTZAN FEATURE DATASET")
    print("=" * 60)

    X, y = build_feature_dataset(
        dataset_path
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        "X shape:",
        X.shape
    )

    print(
        "y shape:",
        y.shape
    )

    # --------------------------------------------------------
    # Check if features were extracted
    # --------------------------------------------------------

    if len(X) == 0:

        print()
        print("ERROR: No features were extracted.")
        print("Please check the dataset.")
        exit()

    # --------------------------------------------------------
    # Encode genre labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(
        y
    )

    print()
    print(
        "Encoded labels:",
        label_encoder.classes_
    )

    # --------------------------------------------------------
    # Save feature dataset
    # --------------------------------------------------------

    np.save(
        os.path.join(
            results_path,
            "X_features.npy"
        ),
        X
    )

    np.save(
        os.path.join(
            results_path,
            "y_labels.npy"
        ),
        y_encoded
    )

    np.save(
        os.path.join(
            results_path,
            "genre_classes.npy"
        ),
        label_encoder.classes_
    )

    # --------------------------------------------------------
    # Save sample spectrogram
    # --------------------------------------------------------

    sample_audio = os.path.join(
        dataset_path,
        "blues",
        "blues.00000.wav"
    )

    sample_output = os.path.join(
        results_path,
        "sample_spectrogram.png"
    )

    if os.path.isfile(sample_audio):

        save_sample_spectrogram(
            sample_audio,
            sample_output
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print(
        "results/X_features.npy"
    )

    print(
        "results/y_labels.npy"
    )

    print(
        "results/genre_classes.npy"
    )

    print(
        "results/sample_spectrogram.png"
    )

    print()
    print("Feature extraction completed successfully!")