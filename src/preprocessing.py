import os
import librosa


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


def load_audio(file_path, sample_rate=22050):
    """
    Load an audio file and resample it to the target sample rate.
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr


def get_audio_files(dataset_path):
    """
    Collect audio file paths and their corresponding genre labels.

    Expected structure:

    dataset/GTZAN/
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

    audio_files = []
    labels = []

    for genre in GENRES:
        genre_path = os.path.join(dataset_path, genre)

        if not os.path.isdir(genre_path):
            print(f"Warning: folder not found: {genre_path}")
            continue

        for filename in os.listdir(genre_path):
            if filename.lower().endswith((".wav", ".au", ".mp3")):
                file_path = os.path.join(genre_path, filename)

                audio_files.append(file_path)
                labels.append(genre)

    return audio_files, labels


if __name__ == "__main__":
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "dataset",
        "GTZAN"
    )

    files, labels = get_audio_files(dataset_path)

    print(f"Dataset path: {dataset_path}")
    print(f"Total audio files: {len(files)}")
    print(f"Total labels: {len(labels)}")

    if files:
        print(f"First file: {files[0]}")
        print(f"First label: {labels[0]}")