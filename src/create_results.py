import os
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

    models = ["SVM", "k-NN"]
    test_accuracies = [0.6867, 0.5000]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        test_accuracies
    )

    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("Traditional ML Model Comparison")

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

    output_file = os.path.join(
        results_path,
        "accuracy_comparison.png"
    )

    plt.savefig(output_file)
    plt.close()

    print("Saved:", output_file)