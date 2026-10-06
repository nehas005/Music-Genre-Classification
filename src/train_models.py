import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(results_path):
    """
    Load the extracted feature dataset.
    """

    X = np.load(
        os.path.join(results_path, "X_features.npy")
    )

    y = np.load(
        os.path.join(results_path, "y_labels.npy")
    )

    classes = np.load(
        os.path.join(results_path, "genre_classes.npy")
    )

    return X, y, classes


def split_data(X, y):
    """
    Split data into:
        70% training
        15% validation
        15% testing
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def train_svm(X_train, y_train):
    """
    Train an SVM classifier.
    """

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_knn(X_train, y_train):
    """
    Train a k-NN classifier.
    """

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X, y, model_name):
    """
    Calculate and print model accuracy.
    """

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print()
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":

    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    results_path = os.path.join(
        project_root,
        "results"
    )

    models_path = os.path.join(
        project_root,
        "models"
    )

    os.makedirs(
        models_path,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load features
    # --------------------------------------------------

    print("Loading extracted features...")

    X, y, classes = load_data(
        results_path
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --------------------------------------------------
    # Train / validation / test split
    # --------------------------------------------------

    print()
    print("Splitting dataset...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------
    # Feature scaling
    # --------------------------------------------------

    print()
    print("Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------
    # SVM
    # --------------------------------------------------

    print()
    print("Training SVM...")

    svm_model = train_svm(
        X_train_scaled,
        y_train
    )

    svm_val_accuracy = evaluate_model(
        svm_model,
        X_val_scaled,
        y_val,
        "SVM Validation"
    )

    svm_test_accuracy = evaluate_model(
        svm_model,
        X_test_scaled,
        y_test,
        "SVM Test"
    )

    # --------------------------------------------------
    # k-NN
    # --------------------------------------------------

    print()
    print("Training k-NN...")

    knn_model = train_knn(
        X_train_scaled,
        y_train
    )

    knn_val_accuracy = evaluate_model(
        knn_model,
        X_val_scaled,
        y_val,
        "k-NN Validation"
    )

    knn_test_accuracy = evaluate_model(
        knn_model,
        X_test_scaled,
        y_test,
        "k-NN Test"
    )

    # --------------------------------------------------
    # Save models + scaler
    # --------------------------------------------------

    joblib.dump(
        svm_model,
        os.path.join(
            models_path,
            "svm_model.pkl"
        )
    )

    joblib.dump(
        knn_model,
        os.path.join(
            models_path,
            "knn_model.pkl"
        )
    )

    joblib.dump(
        scaler,
        os.path.join(
            models_path,
            "scaler.pkl"
        )
    )

    # Save test data for later evaluation
    np.save(
        os.path.join(
            results_path,
            "X_test.npy"
        ),
        X_test_scaled
    )

    np.save(
        os.path.join(
            results_path,
            "y_test.npy"
        ),
        y_test
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"SVM test accuracy : {svm_test_accuracy:.4f}"
    )

    print(
        f"k-NN test accuracy: {knn_test_accuracy:.4f}"
    )

    print()
    print("Saved models:")
    print("models/svm_model.pkl")
    print("models/knn_model.pkl")
    print("models/scaler.pkl")