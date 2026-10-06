import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


if __name__ == "__main__":

    project_root = os.path.dirname(
        os.path.dirname(__file__)
    )

    results_path = os.path.join(
        project_root,
        "results"
    )

    os.makedirs(
        results_path,
        exist_ok=True
    )

    # --------------------------------------------------
    # Final test accuracies
    # --------------------------------------------------

    models = [
        "SVM",
        "k-NN",
        "CNN"
    ]

    test_accuracies = [
        0.6867,
        0.5000,
        0.5733
    ]

    # --------------------------------------------------
    # Create comparison graph
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        test_accuracies
    )

    plt.ylim(0, 1)

    plt.ylabel(
        "Test Accuracy"
    )

    plt.xlabel(
        "Machine Learning Model"
    )

    plt.title(
        "Music Genre Classification Model Comparison"
    )

    # Display accuracy above each bar
    for bar, accuracy in zip(
        bars,
        test_accuracies
    ):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.02,
            f"{accuracy:.2%}",
            ha="center"
        )

    plt.tight_layout()

    # --------------------------------------------------
    # Save graph
    # --------------------------------------------------

    output_file = os.path.join(
        results_path,
        "accuracy_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        "SVM  :", f"{0.6867:.2%}"
    )

    print(
        "k-NN :", f"{0.5000:.2%}"
    )

    print(
        "CNN  :", f"{0.5733:.2%}"
    )

    print()
    print("Best Model: SVM")
    print()
    print("Saved:", output_file)