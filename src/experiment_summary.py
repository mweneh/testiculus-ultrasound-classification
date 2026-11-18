"""Summarize frozen Colab results without loading models or dataset images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from src.threshold_config import DEFAULT_THRESHOLD_CONFIG, load_threshold_configuration


DEFAULT_TEST_RESULTS = DEFAULT_THRESHOLD_CONFIG.with_name("final_test_results.json")
DEFAULT_VERIFICATION = DEFAULT_THRESHOLD_CONFIG.with_name("artifact_verification.json")
TEST_METRICS = (
    "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
    "recall_sensitivity", "specificity", "f1",
)


def _close(reported: float, actual: float, decimals: int) -> bool:
    return math.isclose(reported, actual, rel_tol=0, abs_tol=0.5 * 10 ** -decimals + 1e-12)


def load_experiment_results(
    results_path: Path = DEFAULT_TEST_RESULTS,
    threshold_config: Path = DEFAULT_THRESHOLD_CONFIG,
) -> tuple[dict, dict]:
    """Cross-check supplied test metrics against counts and frozen thresholds."""
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    validation = load_threshold_configuration(threshold_config)
    if results["format_version"] != 1 or results["research_experiments_frozen"] is not True:
        raise ValueError("Expected frozen version-1 experiment results")
    if results["partition"] != "test" or results["positive_class"] != 1:
        raise ValueError("Expected test results with positive class 1")
    counts = results["class_counts"]
    n0, n1 = counts["homogeneous"], counts["inhomogeneous"]
    if any(type(n) is not int or n < 1 for n in (n0, n1)):
        raise ValueError("Both test class counts must be positive integers")
    n = n0 + n1
    if results["n_samples"] != n:
        raise ValueError("Test sample count disagrees with class counts")
    decimals = results["reported_metric_decimal_places"]
    if type(decimals) is not int or not 0 <= decimals <= 10:
        raise ValueError("Invalid metric decimal precision")
    if not _close(results["majority_class_baseline_accuracy"], max(n0, n1) / n, decimals):
        raise ValueError("Majority-class baseline disagrees with counts")
    if set(results["models"]) != set(validation["models"]):
        raise ValueError("Test and validation model names disagree")
    for name, record in results["models"].items():
        if record["threshold"] != validation["models"][name]["selected_threshold"]:
            raise ValueError(f"{name}: test threshold differs from frozen validation selection")
        matrix = record["confusion_matrix"]
        rows = matrix["rows_actual_columns_predicted"]
        if matrix["labels"] != [0, 1] or len(rows) != 2 or any(len(row) != 2 for row in rows):
            raise ValueError(f"{name}: expected a binary confusion matrix ordered [0, 1]")
        if any(type(value) is not int or value < 0 for row in rows for value in row):
            raise ValueError(f"{name}: confusion counts must be nonnegative integers")
        (tn, fp), (fn, tp) = rows
        if tn + fp != n0 or fn + tp != n1:
            raise ValueError(f"{name}: confusion matrix class counts disagree")
        recomputed = {
            "accuracy": (tn + tp) / n,
            "balanced_accuracy": (tp / n1 + tn / n0) / 2,
            "recall_sensitivity": tp / n1,
            "specificity": tn / n0,
            "f1": 2 * tp / (2 * tp + fp + fn),
        }
        for key in TEST_METRICS:
            value = record["metrics"][key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name}: invalid {key}")
            if key in recomputed and not _close(value, recomputed[key], decimals):
                raise ValueError(f"{name}: {key} disagrees with confusion matrix")
    return results, validation


def verify_metrics_artifact(results: dict, model_type: str, artifact_path: Path) -> dict:
    """Compare an original src.evaluation metrics.json to the rounded summary.

    This never evaluates images, selects models, or searches thresholds.
    Successful verification is returned with the artifact hash, not written
    back into the frozen user-supplied record.
    """
    raw = Path(artifact_path).read_bytes()
    artifact = json.loads(raw)
    record = results["models"][model_type]
    metrics = artifact["metrics"]
    if artifact["model_type"] != model_type or artifact["n_samples"] != results["n_samples"]:
        raise ValueError(f"{model_type}: artifact model or sample count disagrees")
    if metrics["positive_class"] != 1 or metrics["threshold"] != record["threshold"]:
        raise ValueError(f"{model_type}: artifact positive class or threshold disagrees")
    for key in TEST_METRICS:
        value = metrics[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not _close(
            record["metrics"][key], value, results["reported_metric_decimal_places"]
        ):
            raise ValueError(f"{model_type}: artifact {key} disagrees")
    for key, value in record["confusion_matrix"].items():
        if metrics["confusion_matrix"][key] != value:
            raise ValueError(f"{model_type}: artifact confusion matrix disagrees")
    return {"verified": True, "sha256": hashlib.sha256(raw).hexdigest()}


def load_verification_report(path: Path = DEFAULT_VERIFICATION) -> dict | None:
    """Use aggregate verification evidence only when it matches both frozen records."""
    path = Path(path)
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("verified") is not True or report.get("format_version") != 1:
        raise ValueError("Expected a successful version-1 artifact verification report")
    for field, source in (
        ("threshold_configuration_sha256", DEFAULT_THRESHOLD_CONFIG),
        ("user_supplied_summary_sha256", DEFAULT_TEST_RESULTS),
    ):
        if report[field] != hashlib.sha256(source.read_bytes()).hexdigest():
            raise ValueError(f"Verification report {field} is stale")
    return report


def render_summary(results: dict, validation: dict, verification: dict | None = None) -> str:
    lines = [
        "# Frozen experiment summary", "",
        "Both models completed training, validation threshold selection and final",
        "held-out test evaluation in Google Colab. Experiments are frozen.", "",
        "Test numbers below were supplied by the project developer, rounded to four",
        "decimal places. Count-based metrics agree with the supplied confusion",
        "matrices at that precision. The original rounded summary is preserved",
        "with its status at import; subsequent verification evidence is separate.", "",
    ]
    if verification is None:
        lines += [
            "Original Colab artifacts remain unverified. ROC-AUC and average precision",
            "cannot be reconstructed from confusion matrices alone.", "",
        ]
    else:
        lines += [
            "Original test metric JSONs and prediction CSVs have been cross-checked.",
            "All full-precision test metrics, including ROC-AUC and average precision,",
            "reproduce from the saved probabilities at the frozen thresholds. Both",
            "models have identical test paths and labels. No held-out image inference",
            "or training was performed. Source hashes and complete original metrics",
            "are preserved in `reports/artifact_verification.json`.", "",
            "Both original validation metric JSONs record majority baseline 0.8093023",
            "(696/860), rather than the partition's 0.8062016 (104/129). Original files",
            "are preserved and this discrepancy is flagged. Model metrics and the",
            "frozen threshold selection reproduce; the threshold JSON and final test",
            "artifacts already contain the correct partition baseline.", "",
        ]
    lines += [
        "## Validation: threshold selection", "",
        "Source: `reports/validation_thresholds.json` (full precision, unchanged).",
        "Maximize positive-class F1 over unique validation probabilities; use `>=`",
        "and choose the highest threshold for exact F1 ties. Test results were not",
        "used for this selection. Validation has 129 images (104 class 0, 25 class 1).", "",
        "| Model | Frozen threshold | Validation precision | Sensitivity | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in validation["models"].items():
        m = record["metrics_at_selected_threshold"]
        lines.append(f"| {name} | {record['selected_threshold']} | {m['precision']:.4f} | "
                     f"{m['recall_sensitivity']:.4f} | {m['specificity']:.4f} | {m['f1']:.4f} |")
    if verification is not None:
        lines += [
            "", "Validation ranking metrics from the original prediction files",
            "(independent of the classification threshold):", "",
            "| Model | Validation ROC-AUC | Validation average precision |",
            "|---|---:|---:|",
        ]
        for name, model in verification["models"].items():
            metrics = model["validation_predictions"]["metrics"]
            lines.append(f"| {name} | {metrics['roc_auc']:.4f} | {metrics['average_precision']:.4f} |")
        lines += [
            "", "Training history and selected checkpoints agree:", "",
            "| Model | Epochs completed | Selected epoch | Best validation loss |",
            "|---|---:|---:|---:|",
        ]
        for name, model in verification["models"].items():
            metadata = model["checkpoint_metadata"]
            lines.append(f"| {name} | {model['epochs_completed']} | {metadata['best_epoch']} | "
                         f"{metadata['best_validation_loss']:.8f} |")
    lines += [
        "", "## Test: final frozen evaluation", "",
        "Both models used the same 129-image held-out set: 104 homogeneous (0),",
        "25 inhomogeneous (positive class 1). Thresholds were fixed from validation.", "",
        "| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in results["models"].items():
        values = " | ".join(f"{record['metrics'][key]:.4f}" for key in TEST_METRICS)
        lines.append(f"| {name} | {values} |")
    lines += ["", "Confusion matrices: rows = actual [0, 1], columns = predicted [0, 1].", ""]
    for name, record in results["models"].items():
        lines.append(f"- {name}: `{record['confusion_matrix']['rows_actual_columns_predicted']}`.")
    lines += [
        "", "Majority-class accuracy baseline: 104 / 129 = **80.62%**.",
        "Both model accuracies are below this baseline. An always-homogeneous",
        "classifier has zero positive-class sensitivity and balanced accuracy 0.5.",
        "ResNet18 has higher reported ROC-AUC, average precision, balanced accuracy,",
        "sensitivity and F1; this comparison does not establish statistical significance.", "",
        "## Reproduction and limits", "",
        "`python -m src.experiment_summary --output reports/experiment_summary.md`",
        "regenerates this summary from the frozen JSON records, without opening",
        "images, loading models, or reading test manifests. Supply original",
        "`src.evaluation` metrics.json files using `--custom-cnn-metrics` and/or",
        "`--resnet18-metrics` to cross-check them and print their SHA-256 hashes.",
        "`python -m src.artifact_verification --artifact-root . --data-dir data/raw",
        "--check-api --output reports/artifact_verification.json` verifies downloaded",
        "artifacts and trained-model inference, sampling validation images only.",
        "Omit `--data-dir` to use synthetic inputs instead of medical images.",
        "Checkpoint and artifact hashes fingerprint the supplied local copies; no",
        "independent provider checksum is available. The original Colab package/",
        "runtime versions remain unknown; the verification environment is recorded.", "",
        "Patient and acquisition-centre identifiers are unavailable. This is a",
        "provisional image-level evaluation, not verified patient-independent or",
        "centre-held-out validation. No clinical validation or diagnostic suitability",
        "is established. Grad-CAM visualizations do not establish localization truth.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_TEST_RESULTS)
    parser.add_argument("--threshold-config", type=Path, default=DEFAULT_THRESHOLD_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--custom-cnn-metrics", type=Path)
    parser.add_argument("--resnet18-metrics", type=Path)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args()
    results, validation = load_experiment_results(args.results, args.threshold_config)
    checks = {}
    for name, path in (("custom_cnn", args.custom_cnn_metrics), ("resnet18", args.resnet18_metrics)):
        if path is not None:
            checks[name] = verify_metrics_artifact(results, name, path)
    if checks:
        print(json.dumps({"original_metrics_verification": checks}, indent=2))
    summary = render_summary(results, validation, load_verification_report(args.verification))
    if args.output is None:
        print(summary, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
