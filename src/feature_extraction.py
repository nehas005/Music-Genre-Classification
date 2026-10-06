import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


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


def extract_mel_spectrogram(
    file_path,
    sample_rate=22050,
    n_mels=128,
    n_fft=2048,
    hop_length=512
):
    """
    Extract a log-scaled mel-spectrogram from an audio file.
    """

    audio, sr = librosa.load(file_path, sr=sample_rate)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def spectrogram_to_features(mel_spectrogram):
    """
    Convert a mel-spectrogram into a fixed-length feature vector.

    We calculate the mean and standard deviation across time
    for every mel-frequency band.
    """

    mean_features = np.mean(mel_spectrogram, axis=1)
    std_features = np.std(mel_spectrogram, axis=1)

    features = np.concatenate(
        [mean_features, std_features]
    )

    return features


def build_feature_dataset(dataset_path):
    """
    Process all GTZAN audio files and create X and y.

    X shape:
        (number_of_audio_files, number_of_features)

    y shape:
        (number_of_audio_files,)
    """

    X = []
    y = []

    total_files = 0

    for genre in GENRES:

        genre_path = os.path.join(
            dataset_path,
            genre
        )

        if not os.path.isdir(genre_path):
            print(f"Warning: missing folder: {genre_path}")
            continue

        audio_files = [
            f for f in os.listdir(genre_path)
            if f.lower().endswith((".wav", ".au", ".mp3"))
        ]

        print(f"{genre}: {len(audio_files)} files")

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

                total_files += 1

            except Exception as e:
                print(
                    f"Error processing {file_path}: {e}"
                )

    X = np.array(X)
    y = np.array(y)

    return X, y


def save_sample_spectrogram(
    audio_file,
    output_file
):
    """
    Save a visual example of a mel-spectrogram.
    """

    mel = extract_mel_spectrogram(
        audio_file
    )

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mel,
        sr=22050,
        hop_length=512,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-Spectrogram")
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


if __name__ == "__main__":

    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    dataset_path = os.path.join(
        project_root,
        "dataset",
        "GTZAN"
    )

    results_path = os.path.join(
        project_root,
        "results"
    )

    os.makedirs(
        results_path,
        exist_ok=True
    )

    print("=" * 60)
    print("Building GTZAN feature dataset")
    print("=" * 60)

    X, y = build_feature_dataset(
        dataset_path
    )

    print()
    print("=" * 60)
    print("Feature extraction complete")
    print("=" * 60)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Encode genre names as integers.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Encoded labels:", label_encoder.classes_)

    # Save feature arrays.
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

    # Save genre names.
    np.save(
        os.path.join(
            results_path,
            "genre_classes.npy"
        ),
        label_encoder.classes_
    )

    print()
    print("Saved:")
    print("results/X_features.npy")
    print("results/y_labels.npy")
    print("results/genre_classes.npy")