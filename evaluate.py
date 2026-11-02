"""Evaluate a trained SmartRupee model on the held-out test dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow import keras


SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    return sum(
        1 for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def combine_history(history):
    combined = {}
    for stage in (history.get("initial_training", {}), history.get("fine_tuning", {})):
        for metric, values in stage.items():
            combined.setdefault(metric, []).extend(values)
    return combined


def plot_history(history, output_path):
    combined = combine_history(history)
    if not any(metric in combined for metric in ("accuracy", "val_accuracy", "loss", "val_loss")):
        return False

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(combined.get("loss", combined.get("accuracy", []))) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric_name, validation_name, title in (
        (axes[0], "accuracy", "val_accuracy", "Accuracy"),
        (axes[1], "loss", "val_loss", "Loss"),
    ):
        if metric_name in combined:
            axis.plot(combined[metric_name], label=f"Training {title.lower()}")
        if validation_name in combined:
            axis.plot(combined[validation_name], label=f"Validation {title.lower()}")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.legend()
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def overfitting_assessment(history):
    combined = combine_history(history)
    assessment = {"status": "insufficient_history", "indicators": []}
    if not combined:
        return assessment

    train_accuracy = combined.get("accuracy", [])
    val_accuracy = combined.get("val_accuracy", [])
    train_loss = combined.get("loss", [])
    val_loss = combined.get("val_loss", [])
    if train_accuracy and val_accuracy and train_loss and val_loss:
        accuracy_gap = float(train_accuracy[-1] - val_accuracy[-1])
        loss_gap = float(val_loss[-1] - train_loss[-1])
        if accuracy_gap >= 0.10 or loss_gap >= 0.25:
            status = "possible_overfitting"
        elif accuracy_gap <= 0.03 and loss_gap <= 0.10:
            status = "no_strong_overfitting_signal"
        else:
            status = "possible_overfitting_need_caution"
        assessment.update({
            "status": status,
            "final_training_accuracy": float(train_accuracy[-1]),
            "final_validation_accuracy": float(val_accuracy[-1]),
            "accuracy_gap": accuracy_gap,
            "final_training_loss": float(train_loss[-1]),
            "final_validation_loss": float(val_loss[-1]),
            "validation_minus_training_loss": loss_gap,
        })
    return assessment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--model", default="models/currency_model.keras")
    parser.add_argument("--class-names", default="models/class_names.json")
    parser.add_argument("--history", default="reports/training_history.json")
    args = parser.parse_args()

    model_path = Path(args.model)
    names_path = Path(args.class_names)
    history_path = Path(args.history)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}. Train the model before evaluation.")
    if not names_path.is_file():
        raise SystemExit(f"Class names not found: {names_path}. Train the model before evaluation.")

    with names_path.open("r", encoding="utf-8") as handle:
        class_names = json.load(handle)

    test_root = Path(args.data) / "test"
    test_counts = {
        class_name: count_images(test_root / class_name)
        for class_name in class_names
    }
    if not sum(test_counts.values()):
        raise SystemExit(
            f"No test images found under {test_root}. Add held-out labelled images before evaluation."
        )
    missing_classes = [name for name, count in test_counts.items() if count == 0]
    if missing_classes:
        raise SystemExit(
            "Test split is missing images for class(es): " + ", ".join(missing_classes)
        )

    test_dataset = keras.utils.image_dataset_from_directory(
        test_root,
        image_size=SIZE,
        batch_size=16,
        shuffle=False,
        label_mode="int",
        class_names=class_names,
    )

    true_labels = np.concatenate([labels.numpy() for _, labels in test_dataset])
    predictions = np.argmax(keras.models.load_model(model_path).predict(test_dataset, verbose=0), axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=range(len(class_names)), zero_division=0
    )
    report = classification_report(
        true_labels, predictions, target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(true_labels, predictions, labels=range(len(class_names)))

    counts = {}
    imbalance = {}
    for class_name in class_names:
        counts[class_name] = count_images(test_root / class_name)
    majority = max(counts.values(), default=0)
    minority = min(counts.values(), default=0)
    for class_name, count in counts.items():
        imbalance[class_name] = {
            "count": count,
            "share": count / max(sum(counts.values()), 1),
        }

    history = {}
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)

    plots_saved = plot_history(history, reports_dir / "training_validation.png")
    result = {
        "dataset": str(test_root),
        "model": str(model_path),
        "classes": class_names,
        "n_test": int(len(true_labels)),
        "class_counts": counts,
        "class_imbalance": {
            "counts": imbalance,
            "majority_to_minority_ratio": float(majority / max(minority, 1)),
            "is_imbalanced": minority > 0 and majority / minority >= 1.5,
        },
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(class_names)
        },
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "overfitting": overfitting_assessment(history),
        "plots": {
            "training_validation": str(reports_dir / "training_validation.png") if plots_saved else None,
        },
        "limitations": [
            "Results are valid only for the dataset and note variants represented in this test split.",
            "Report them explicitly as held-out performance; do not claim real-world currency reliability.",
            "Verify provenance, capture conditions, denomination coverage, printer/print-run diversity, and anti-leakage checks.",
            "A small or highly similar dataset can make held-out accuracy optimistic and increase overfitting risk.",
            "Class imbalance, location leakage, sensor differences, lighting, blur, and counterfeit sophistication can bias results.",
        ],
    }

    output_path = reports_dir / "evaluation.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(classification_report(true_labels, predictions, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print("Overfitting assessment:", result["overfitting"]["status"])
    print("Saved:", output_path)
    if plots_saved:
        print("Saved:", reports_dir / "training_validation.png")


if __name__ == "__main__":
    main()
