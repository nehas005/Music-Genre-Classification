# 🎵 Music Genre Classification

A machine learning and deep learning project for automatically classifying music into different genres using the **GTZAN Music Genre Dataset**.

The project compares traditional machine learning models (**SVM and k-NN**) with a **Convolutional Neural Network (CNN)** using audio features and Mel-spectrogram representations.

---

## 📌 Project Overview

Music genre classification is the task of automatically identifying the genre of a music recording based on its audio characteristics.

In this project, audio files are processed to extract meaningful features from their frequency information. These features are then used to train and evaluate multiple classification models.

The project implements:

- Audio preprocessing
- Mel-spectrogram generation
- Statistical feature extraction
- Feature scaling
- Dataset splitting
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- Convolutional Neural Network (CNN)
- Model evaluation
- Confusion matrix analysis
- Accuracy comparison

The objective is to compare traditional machine learning approaches with a deep learning approach and determine which model performs best on the test dataset.

---

# 📂 Dataset

The project uses the **GTZAN Music Genre Dataset**.

The dataset contains music belonging to 10 different genres:

1. Blues
2. Classical
3. Country
4. Disco
5. Hip-Hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock

Each genre is represented as a separate class.

During feature extraction, one corrupted audio file:

`jazz.00054.wav`

was excluded because it could not be decoded successfully.

After removing the corrupted file, the project contains:

**999 usable audio samples.**

---

# 🎯 Objectives

The main objectives of this project are:

- To preprocess music audio files.
- To convert audio signals into Mel-spectrogram representations.
- To extract useful numerical features from the spectrograms.
- To train traditional machine learning classifiers.
- To train a CNN using spectrogram-based representations.
- To evaluate the performance of each model.
- To compare the models using test accuracy.
- To analyze classification errors using confusion matrices.
- To identify the best-performing model.

---

# 🛠️ Technologies Used

The project is implemented using Python and the following technologies:

- **Python**
- **NumPy**
- **Librosa**
- **Scikit-learn**
- **TensorFlow / Keras**
- **Matplotlib**
- **Pandas**

### Machine Learning

- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)

### Deep Learning

- Convolutional Neural Network (CNN)

### Audio Processing

- Mel-spectrogram
- Log-scaled Mel-spectrogram
- Statistical feature extraction

---

# 🔄 Project Workflow

The overall workflow of the project is:

```text
             GTZAN Audio Dataset
                     │
                     ▼
             Audio Preprocessing
                     │
                     ▼
             Mel-Spectrogram
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
   Feature Extraction      Spectrogram Input
          │                     │
          ▼                     ▼
   Mean + Standard Dev.        CNN
          │                     │
          ▼                     │
     256 Features               │
          │                     │
          ▼                     │
    Feature Scaling             │
          │                     │
       ┌──┴───┐                 │
       ▼      ▼                 │
      SVM    k-NN               │
       │      │                 │
       └──┬───┘                 │
          │                     │
          └─────────┬───────────┘
                    ▼
              Model Evaluation
                    │
                    ▼
          Accuracy + Confusion
                Matrices
