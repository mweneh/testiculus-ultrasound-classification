"""Select classification thresholds from saved validation predictions only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluation import calculate_metrics
from src.threshold_config import (
    METRIC_KEYS,
    MODEL_NAMES,
    SELECTION_CRITERIA,
    load_threshold_configuration,
    validate_threshold_configuration,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_validation_predictions(
    predictions_path: Path,
    validation_manifest_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load predictions only when their paths and labels exactly match validation."""
    manifest_rows = _read_csv(validation_manifest_path)
    prediction_rows = _read_csv(predictions_path)
    required_manifest = {"image_path", "label"}
    required_predictions = {"image_path", "label", "probability_class_1"}

    if not manifest_rows or not required_manifest.issubset(manifest_rows[0]):
        raise ValueError("Validation manifest must contain image_path and label")
    if not prediction_rows or not required_predictions.issubset(prediction_rows[0]):
        raise ValueError(
            "Prediction CSV must contain image_path, label, and probability_class_1"
        )

    def index_rows(
        rows: list[dict[str, str]], description: str
    ) -> dict[str, dict[str, str]]:
        indexed = {}
        for row in rows:
            image_path = Path(row["image_path"])
            if image_path.is_absolute() or ".." in image_path.parts:
                raise ValueError(f"{description} contains an invalid relative image path")
            key = image_path.as_posix()
            if key in indexed:
                raise ValueError(f"{description} contains duplicate image paths")
            indexed[key] = row
        return indexed

    expected = index_rows(manifest_rows, "Validation manifest")
    actual = index_rows(prediction_rows, "Prediction CSV")
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        unexpected = sorted(actual.keys() - expected.keys())
        raise ValueError(
            "Prediction paths must match the validation manifest exactly "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    labels = []
    probabilities = []
    for image_path, expected_row in expected.items():
        prediction_row = actual[image_path]
        try:
            expected_label = int(expected_row["label"])
            label = int(prediction_row["label"])
            probability = float(prediction_row["probability_class_1"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid label or probability for {image_path}") from error
        if label != expected_label:
            raise ValueError(
                f"Prediction label disagrees with validation manifest: {image_path}"
            )
        if label not in (0, 1):
            raise ValueError(f"Validation label must be binary: {image_path}")
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid positive-class probability for {image_path}")
        labels.append(label)
        probabilities.append(probability)

    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def threshold_sweep(labels, probabilities) -> list[dict[str, float | None]]:
    """Evaluate positive-class metrics at every distinct predicted score."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("labels and probabilities must be equal-length vectors")
    if labels.size == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("A non-empty binary validation label vector is required")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    if not np.any(labels == 1):
        raise ValueError("Positive-class F1 threshold selection requires positive examples")

    sweep = []
    for threshold in np.unique(probabilities):
        metrics = calculate_metrics(labels, probabilities, threshold=float(threshold))
        sweep.append({
            "threshold": float(threshold),
            "precision": metrics["precision"],
            "recall": metrics["recall_sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
        })
    return sweep


def _requested_metrics(labels, probabilities, threshold: float) -> dict:
    metrics = calculate_metrics(labels, probabilities, threshold=threshold)
    return {key: metrics[key] for key in METRIC_KEYS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_threshold_configuration(
    prediction_files: dict[str, Path],
    validation_manifest_path: Path,
    output_path: Path,
) -> dict:
    """Create a new validation record, refusing to overwrite an experiment."""
    if set(prediction_files) != set(MODEL_NAMES):
        raise ValueError(f"Provide predictions for exactly: {', '.join(MODEL_NAMES)}")

    models = {}
    for model_name in MODEL_NAMES:
        predictions_path = Path(prediction_files[model_name])
        labels, probabilities = load_validation_predictions(
            predictions_path, validation_manifest_path
        )
        sweep = threshold_sweep(labels, probabilities)
        # Highest threshold wins exact F1 ties, reducing the predicted-positive
        # set while preserving the maximum validation F1.
        selected = max(sweep, key=lambda row: (row["f1"], row["threshold"]))
        selected_threshold = float(selected["threshold"])
        models[model_name] = {
            "predictions_file": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "n_validation_samples": int(labels.size),
            "majority_class_baseline_accuracy": float(
                np.bincount(labels, minlength=2).max() / labels.size
            ),
            "selected_threshold": selected_threshold,
            "metrics_at_0.5": _requested_metrics(labels, probabilities, 0.5),
            "metrics_at_selected_threshold": _requested_metrics(
                labels, probabilities, selected_threshold
            ),
            "threshold_sweep": sweep,
        }

    configuration = {
        "format_version": 1,
        "validation_manifest": str(validation_manifest_path),
        "validation_manifest_sha256": _sha256(validation_manifest_path),
        "selection_criteria": dict(SELECTION_CRITERIA),
        "models": models,
    }
    validate_threshold_configuration(configuration)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(configuration, indent=2, allow_nan=False) + "\n")
    return configuration


def verify_threshold_configuration(
    configuration_path: Path,
    validation_manifest_path: Path,
    prediction_files: dict[str, Path] | None = None,
) -> dict:
    """Read-only verification against an explicit validation manifest and CSVs.

    Source paths can differ from the recorded Colab paths. Require their exact
    recorded byte hashes before recomputing the complete sweep and metrics.
    No images or other partitions are opened. Missing sources are reported as
    unverified, never treated as a successful prediction reproduction.
    """
    configuration = load_threshold_configuration(configuration_path)
    if _sha256(validation_manifest_path) != configuration["validation_manifest_sha256"]:
        raise ValueError("Validation manifest SHA-256 does not match the record")
    rows = _read_csv(validation_manifest_path)
    try:
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid validation manifest labels") from error
    if not labels.size or not np.isin(labels, (0, 1)).all() or not np.any(labels == 1):
        raise ValueError("Validation manifest requires binary labels and positive examples")
    baseline = float(np.bincount(labels, minlength=2).max() / labels.size)
    prediction_files = {} if prediction_files is None else prediction_files
    if not set(prediction_files).issubset(MODEL_NAMES):
        raise ValueError(f"Unsupported prediction model; expected {MODEL_NAMES}")
    result = {"validation_manifest_sha256_verified": True, "models": {}}
    for name, record in configuration["models"].items():
        if record["n_validation_samples"] != labels.size:
            raise ValueError(f"{name}: validation sample count does not match")
        if record["majority_class_baseline_accuracy"] != baseline:
            raise ValueError(f"{name}: validation majority baseline does not match")
        verified = name in prediction_files
        if verified:
            path = Path(prediction_files[name])
            if _sha256(path) != record["predictions_sha256"]:
                raise ValueError(f"{name}: predictions SHA-256 does not match the record")
            source_labels, probabilities = load_validation_predictions(path, validation_manifest_path)
            recomputed = {
                "threshold_sweep": threshold_sweep(source_labels, probabilities),
                "metrics_at_0.5": _requested_metrics(source_labels, probabilities, 0.5),
                "metrics_at_selected_threshold": _requested_metrics(
                    source_labels, probabilities, record["selected_threshold"]
                ),
            }
            for key, value in recomputed.items():
                if record[key] != value:
                    raise ValueError(f"{name}: recomputed {key} does not match the record")
        result["models"][name] = {
            "selected_threshold": record["selected_threshold"],
            "n_validation_samples": record["n_validation_samples"],
            "recorded_sweep_verified": True,
            "source_predictions_verified": verified,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-cnn-predictions", type=Path)
    parser.add_argument("--resnet18-predictions", type=Path)
    parser.add_argument("--verify-config", type=Path, help="Verify an existing record without modifying it")
    parser.add_argument(
        "--validation-manifest", type=Path,
        default=Path("data/splits/validation.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/validation_thresholds.json"),
    )
    args = parser.parse_args()
    prediction_files = {
        name: path for name, path in (
            ("custom_cnn", args.custom_cnn_predictions),
            ("resnet18", args.resnet18_predictions),
        ) if path is not None
    }
    if args.verify_config is not None:
        result = verify_threshold_configuration(
            args.verify_config, args.validation_manifest, prediction_files
        )
        print(json.dumps(result, indent=2, allow_nan=False))
        return
    if set(prediction_files) != set(MODEL_NAMES):
        parser.error("Selection requires both --custom-cnn-predictions and --resnet18-predictions")
    configuration = create_threshold_configuration(
        prediction_files,
        args.validation_manifest,
        args.output,
    )
    print(json.dumps(configuration, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
