import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import joblib
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(__file__)
)

models_path = os.path.join(
    project_root,
    "models"
)

results_path = os.path.join(
    project_root,
    "results"
)


# --------------------------------------------------
# Genre names
# --------------------------------------------------

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
    "rock"
]


# --------------------------------------------------
# Load traditional ML test data
# --------------------------------------------------

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

X_test = np.load(
    os.path.join(
        results_path,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        results_path,
        "y_test.npy"
    )
)

print("Traditional ML test shape:", X_test.shape)
print("Traditional ML labels:", y_test.shape)


# --------------------------------------------------
# Load SVM and k-NN
# --------------------------------------------------

print()
print("=" * 60)
print("LOADING SVM AND k-NN")
print("=" * 60)

svm_model = joblib.load(
    os.path.join(
        models_path,
        "svm_model.pkl"
    )
)

knn_model = joblib.load(
    os.path.join(
        models_path,
        "knn_model.pkl"
    )
)

print("SVM loaded successfully!")
print("k-NN loaded successfully!")


# --------------------------------------------------
# SVM predictions
# --------------------------------------------------

print()
print("=" * 60)
print("SVM EVALUATION")
print("=" * 60)

svm_predictions = svm_model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        svm_predictions,
        target_names=GENRES,
        zero_division=0
    )
)


# --------------------------------------------------
# SVM confusion matrix
# --------------------------------------------------

svm_cm = confusion_matrix(
    y_test,
    svm_predictions
)

plt.figure(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=svm_cm,
    display_labels=GENRES
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "SVM Confusion Matrix"
)

plt.tight_layout()

svm_cm_file = os.path.join(
    results_path,
    "svm_confusion_matrix.png"
)

plt.savefig(
    svm_cm_file,
    dpi=300
)

plt.close()

print(
    "Saved:",
    svm_cm_file
)


# --------------------------------------------------
# k-NN predictions
# --------------------------------------------------

print()
print("=" * 60)
print("k-NN EVALUATION")
print("=" * 60)

knn_predictions = knn_model.predict(
    X_test
)

print(
    classification_report(
        y_test,
        knn_predictions,
        target_names=GENRES,
        zero_division=0
    )
)


# --------------------------------------------------
# k-NN confusion matrix
# --------------------------------------------------

knn_cm = confusion_matrix(
    y_test,
    knn_predictions
)

plt.figure(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=knn_cm,
    display_labels=GENRES
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "k-NN Confusion Matrix"
)

plt.tight_layout()

knn_cm_file = os.path.join(
    results_path,
    "knn_confusion_matrix.png"
)

plt.savefig(
    knn_cm_file,
    dpi=300
)

plt.close()

print(
    "Saved:",
    knn_cm_file
)


# --------------------------------------------------
# Load CNN test data
# --------------------------------------------------

print()
print("=" * 60)
print("LOADING CNN")
print("=" * 60)

cnn_X_test = np.load(
    os.path.join(
        results_path,
        "cnn_X_test.npy"
    )
)

cnn_y_test = np.load(
    os.path.join(
        results_path,
        "cnn_y_test.npy"
    )
)

cnn_model = tf.keras.models.load_model(
    os.path.join(
        models_path,
        "cnn_model.keras"
    )
)

print(
    "CNN test shape:",
    cnn_X_test.shape
)

print(
    "CNN loaded successfully!"
)


# --------------------------------------------------
# CNN predictions
# --------------------------------------------------

print()
print("=" * 60)
print("CNN EVALUATION")
print("=" * 60)

cnn_probabilities = cnn_model.predict(
    cnn_X_test,
    verbose=0
)

cnn_predictions = np.argmax(
    cnn_probabilities,
    axis=1
)

print(
    classification_report(
        cnn_y_test,
        cnn_predictions,
        target_names=GENRES,
        zero_division=0
    )
)


# --------------------------------------------------
# CNN confusion matrix
# --------------------------------------------------

cnn_cm = confusion_matrix(
    cnn_y_test,
    cnn_predictions
)

plt.figure(
    figsize=(9, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cnn_cm,
    display_labels=GENRES
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "CNN Confusion Matrix"
)

plt.tight_layout()

cnn_cm_file = os.path.join(
    results_path,
    "cnn_confusion_matrix.png"
)

plt.savefig(
    cnn_cm_file,
    dpi=300
)

plt.close()

print(
    "Saved:",
    cnn_cm_file
)


# --------------------------------------------------
# Complete
# --------------------------------------------------

print()
print("=" * 60)
print("MODEL EVALUATION COMPLETE")
print("=" * 60)

print("Created files:")

print(
    "results/svm_confusion_matrix.png"
)

print(
    "results/knn_confusion_matrix.png"
)

print(
    "results/cnn_confusion_matrix.png"
)