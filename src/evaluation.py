"""Checkpoint-based evaluation and classification reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data.dataset import TesticulUSDataset
from src.data.transforms import get_transforms
from src.training import load_checkpoint, select_device


def calculate_metrics(labels, probabilities, threshold: float = 0.5,
                      majority_baseline: float | None = None) -> dict:
    """Calculate class-sensitive metrics for inhomogeneous class 1.

    Metrics whose mathematical preconditions are not met are returned as
    ``None`` and listed in ``undefined_metrics``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("labels and probabilities must be equal-length vectors")
    if labels.size == 0:
        raise ValueError("Cannot evaluate an empty label set")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Labels must be binary values 0 or 1")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if majority_baseline is None:
        class_counts = np.bincount(labels, minlength=2)
        majority_baseline = float(class_counts.max() / labels.size)
    elif not np.isfinite(majority_baseline) or not 0 <= majority_baseline <= 1:
        raise ValueError("majority_baseline must be finite and in [0, 1]")

    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    both_classes_present = np.unique(labels).size == 2
    positive_predictions = tp + fp
    positive_actual = tp + fn
    negative_actual = tn + fp
    f1_denominator = 2 * tp + fp + fn

    undefined = []
    if not both_classes_present:
        roc_auc = None
        average_precision = None
        balanced_accuracy = None
        undefined.extend(("roc_auc", "average_precision", "balanced_accuracy"))
    else:
        roc_auc = float(roc_auc_score(labels, probabilities))
        average_precision = float(average_precision_score(labels, probabilities))
        balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    if positive_predictions:
        precision = tp / positive_predictions
    else:
        precision = None
        undefined.append("precision")
    if positive_actual:
        recall = tp / positive_actual
    else:
        recall = None
        undefined.append("recall_sensitivity")
    if negative_actual:
        specificity = tn / negative_actual
    else:
        specificity = None
        undefined.append("specificity")
    if f1_denominator:
        f1 = 2 * tp / f1_denominator
    else:
        f1 = None
        undefined.append("f1")

    return {
        "positive_class": 1,
        "threshold": float(threshold),
        "majority_class_baseline_accuracy": float(majority_baseline),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": {
            "labels": [0, 1],
            "rows_actual_columns_predicted": matrix.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "undefined_metrics": undefined,
    }


def collect_predictions(model: torch.nn.Module, loader: DataLoader,
                        device: str | torch.device = "auto") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return labels, positive-class probabilities, and class-qualified paths."""
    target_device = select_device(device)
    model.to(target_device)
    model.eval()
    labels_all = []
    probabilities_all = []
    paths_all = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(target_device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(batch["label"].cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            paths_all.extend(batch["image_path"])
    if not labels_all:
        raise ValueError("Cannot evaluate an empty data loader")
    return (
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(probabilities_all, dtype=np.float64),
        paths_all,
    )


def save_roc_pr_plots(labels, probabilities, output_dir: Path) -> tuple[Path, Path]:
    """Save ROC and precision-recall curves, annotating undefined curves."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_path = output_dir / "roc_curve.png"
    pr_path = output_dir / "precision_recall_curve.png"

    if np.unique(labels).size == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
        roc_auc = roc_auc_score(labels, probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(false_positive_rate, true_positive_rate,
                 label=f"ROC-AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        plt.close()

        precision, recall, _ = precision_recall_curve(labels, probabilities)
        average_precision = average_precision_score(labels, probabilities)
        prevalence = float(np.mean(labels == 1))
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision,
                 label=f"Average precision = {average_precision:.3f}")
        plt.axhline(prevalence, linestyle="--", color="gray",
                    label=f"Positive prevalence = {prevalence:.3f}")
        plt.xlabel("Recall / sensitivity")
        plt.ylabel("Precision")
        plt.title("Precision-recall curve")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=150)
        plt.close()
    else:
        for path, title in ((roc_path, "ROC curve"), (pr_path, "Precision-recall curve")):
            plt.figure(figsize=(6, 5))
            plt.text(0.5, 0.5, "Undefined: both classes are required",
                     ha="center", va="center")
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
    return roc_path, pr_path


def evaluate_checkpoint(checkpoint_path: Path, data_dir: Path,
                        manifest_path: Path, output_dir: Path,
                        threshold: float = 0.5,
                        device: str | torch.device = "auto",
                        batch_size: int = 32,
                        num_workers: int = 0) -> dict:
    """Evaluate a selected checkpoint on one explicit manifest."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    target_device = select_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, device=target_device)
    model_type = checkpoint["model_type"]
    dataset = TesticulUSDataset(
        data_dir,
        manifest_path,
        transform=get_transforms(model_type, training=False),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    labels, probabilities, image_paths = collect_predictions(model, loader, target_device)
    metrics = calculate_metrics(labels, probabilities, threshold=threshold)
    predictions = (probabilities >= threshold).astype(np.int64)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("image_path", "label", "prediction", "probability_class_0",
                        "probability_class_1", "threshold"),
            lineterminator="\n",
        )
        writer.writeheader()
        for image_path, label, prediction, positive_probability in zip(
                image_paths, labels, predictions, probabilities):
            writer.writerow({
                "image_path": image_path,
                "label": int(label),
                "prediction": int(prediction),
                "probability_class_0": float(1 - positive_probability),
                "probability_class_1": float(positive_probability),
                "threshold": float(threshold),
            })

    save_roc_pr_plots(labels, probabilities, output_dir)
    result = {
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "n_samples": int(len(labels)),
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    result = evaluate_checkpoint(
        args.checkpoint,
        args.data_dir,
        args.manifest,
        args.output_dir,
        threshold=args.threshold,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
