"""Summarize saved exploratory results and measured efficiency without model execution."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

from src.artifact_verification import verify_prediction_artifact
from src.exploratory_evaluation import check_original_artifacts, sha256


def create_comparison_report() -> dict:
    freeze_path = Path("reports/mobilenet_v3_small_freeze.json")
    frozen = json.loads(freeze_path.read_text())
    protocol = json.loads(Path(frozen["sources"]["protocol"]["path"]).read_text())
    check_original_artifacts(protocol)
    for source in frozen["sources"].values():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Frozen evidence changed")
    test_dir = Path("results/mobilenet_v3_small_test")
    receipt = json.loads((test_dir / "evaluation_receipt.json").read_text())
    if (not receipt["complete"] or receipt["evaluation_attempt"] != 1
            or receipt["freeze_sha256"] != sha256(freeze_path)
            or datetime.fromisoformat(receipt["started_utc"]) <= datetime.fromisoformat(frozen["frozen_before_test_evaluation_utc"])):
        raise ValueError("Require one complete evaluation after the validation freeze")
    for field, name in [("metrics_sha256", "metrics.json"), ("predictions_sha256", "predictions.csv")]:
        if receipt[field] != sha256(test_dir / name):
            raise ValueError("Single-evaluation output hash changed")
    mobile = verify_prediction_artifact(test_dir / "predictions.csv", test_dir / "metrics.json", frozen["threshold"])
    original_path = Path("reports/artifact_verification.json")
    original = json.loads(original_path.read_text())
    benchmark_path = Path("reports/model_efficiency.json")
    benchmark = json.loads(benchmark_path.read_text())
    models = {}
    for name in ("custom_cnn", "resnet18", "mobilenet_v3_small"):
        saved = mobile if name == "mobilenet_v3_small" else original["models"][name]["test_predictions"]
        efficiency = benchmark["models"][name]
        expected_hash = (frozen["sources"]["checkpoint"]["sha256"] if name == "mobilenet_v3_small"
                         else original["models"][name]["checkpoint_sha256"])
        if efficiency["checkpoint_sha256"] != expected_hash or efficiency["weight_status"] != "trained checkpoint":
            raise ValueError("Efficiency benchmark must use the frozen trained checkpoints")
        if saved["sample_paths_and_labels_sha256"] != mobile["sample_paths_and_labels_sha256"]:
            raise ValueError("Models were not evaluated on identical sample paths and labels")
        models[name] = {"comparison_status": "post-hoc exploratory" if name == "mobilenet_v3_small" else "original frozen comparison",
                        "efficiency": efficiency, "test": saved}
    history_path = Path("training_runs/mobilenet_v3_small/history.csv")
    with history_path.open(newline="") as stream:
        history = [{key: int(value) if key == "epoch" else float(value)
                    for key, value in row.items()} for row in csv.DictReader(stream)]
    pretrained = Path("artifacts/torch_cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth")
    return {
        "format_version": 1, "experiment_status": "post-hoc exploratory",
        "original_test_outcomes_previously_observed": True,
        "original_artifacts_unchanged": True, "identical_test_paths_and_labels": True,
        "hardware": benchmark["hardware"], "benchmark_conditions": benchmark["conditions"],
        "benchmark_sha256": sha256(benchmark_path), "original_verification_sha256": sha256(original_path),
        "freeze_sha256": sha256(freeze_path), "freeze": frozen, "evaluation_receipt": receipt,
        "training_history_sha256": sha256(history_path), "training_history": history,
        "pretrained_weights": {"source": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
                               "enum": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
                               "sha256": sha256(pretrained) if pretrained.exists() else None},
        "models": models,
    }


def render_comparison(report: dict) -> str:
    training = report["freeze"]["training"]
    validation = report["freeze"]["validation_metrics_at_frozen_threshold"]
    hardware, conditions = report["hardware"], report["benchmark_conditions"]
    lines = [
        "# Exploratory MobileNetV3-Small comparison", "",
        "MobileNet was added after the original Custom CNN and ResNet18 test results",
        "were known. This is a post-hoc exploratory extension, not an independent",
        "confirmatory comparison. Original models, thresholds and results are unchanged.", "",
        "## CPU training and validation", "",
        f"Device: `{training['device']}`. ImageNet initialization, frozen features and",
        "frozen feature BatchNorm statistics; classification head optimized with AdamW",
        "and training-label class-weighted cross-entropy. Learning rate 0.0001, batch",
        "size 16, maximum 20 epochs, patience 5, seed 42, two CPU threads, no loader workers.",
        f"Completed {training['epochs_completed']} epochs in {training['training_seconds']:.2f} seconds;",
        f"best epoch {training['best_epoch']}, validation loss {training['best_validation_loss']:.6f}.",
        f"Epoch durations: median {statistics.median(training['epoch_seconds']):.2f} seconds,",
        f"range {min(training['epoch_seconds']):.2f}–{max(training['epoch_seconds']):.2f} seconds.",
        "Training duration includes training/validation epochs and checkpoint/history I/O;",
        "it excludes initialization/download and dataset construction. Full epoch metrics",
        "and durations are preserved in `reports/mobilenet_exploratory_results.json`.",
        f"Trainable parameters: {training['trainable_parameters']:,} / {training['parameters']:,}.", "",
        report["freeze"]["fine_tuning_decision"], "",
        "Validation uses 129 images (104 homogeneous, 25 inhomogeneous) and deterministic",
        "224×224 bilinear resizing, grayscale repeated to three channels and ImageNet",
        "normalization. The best checkpoint is selected by minimum validation loss.",
        "Threshold candidates are every unique validation probability; maximize",
        "positive-class F1 and choose the highest threshold on exact ties.", "",
        f"Frozen threshold: `{report['freeze']['threshold']}`.", "",
        "| Validation ROC-AUC | AP | Sensitivity | Specificity | F1 | Balanced accuracy |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {validation['roc_auc']:.4f} | {validation['average_precision']:.4f} | "
        f"{validation['recall_sensitivity']:.4f} | {validation['specificity']:.4f} | "
        f"{validation['f1']:.4f} | {validation['balanced_accuracy']:.4f} |", "",
        "The checkpoint, validation predictions and selected threshold were hashed and",
        "frozen before one guarded test evaluation. Reusing its output directory is",
        "refused, including after a partial failure. No test-based tuning followed.", "",
        "## Measured efficiency and test results", "",
        f"Hardware: {hardware['cpu_model']}; {hardware['platform']}.",
        f"Python {hardware['python_version']}, PyTorch {hardware['torch_version']},",
        f"Torchvision {hardware['torchvision_version']}; CPU float32, batch size 1,",
        f"224×224 inputs, {hardware['intraop_threads']} intra-op / {hardware['interop_threads']} inter-op threads.",
        f"Each model uses {conditions['warmup']} warm-up forwards and {conditions['repeats']} timed forwards.",
        "The same synthetic grayscale tensor is used with each model's channel and",
        "normalization conventions. Latency measures model forward under inference_mode;",
        "image decoding, preprocessing, disk I/O and API overhead are excluded.",
        "State size is the sum of state_dict tensor bytes, including buffers; checkpoint",
        "size is the actual saved file. MiB = 1,048,576 bytes. Timings are hardware-specific.", "",
        "All test metrics use the same saved sample identities (129 images, 104/25).",
        "Rows for the original models reuse verified Colab metrics without rerunning them.", "",
        "| Model | Parameters | State MiB | Checkpoint MiB | Median CPU ms | Test ROC-AUC | Test AP | Sensitivity | Specificity | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    names = {"custom_cnn": "Custom CNN (original)", "resnet18": "ResNet18 (original)",
             "mobilenet_v3_small": "MobileNetV3-Small (exploratory)"}
    for name, model in report["models"].items():
        efficiency, metrics = model["efficiency"], model["test"]["metrics"]
        lines.append(f"| {names[name]} | {efficiency['parameters']:,} | "
                     f"{efficiency['state_dict_bytes']/1048576:.3f} | {efficiency['checkpoint_bytes']/1048576:.3f} | "
                     f"{efficiency['latency']['median_ms']:.3f} | {metrics['roc_auc']:.4f} | "
                     f"{metrics['average_precision']:.4f} | {metrics['recall_sensitivity']:.4f} | "
                     f"{metrics['specificity']:.4f} | {metrics['f1']:.4f} |")
    test = report["models"]["mobilenet_v3_small"]["test"]["metrics"]
    lines += ["", f"Exploratory MobileNet test accuracy: {test['accuracy']:.4f}; balanced accuracy:",
              f"{test['balanced_accuracy']:.4f}; majority-class baseline: {test['majority_class_baseline_accuracy']:.4f}.",
              f"Confusion matrix (actual rows, predicted columns, class order [0,1]): `{test['confusion_matrix']['rows_actual_columns_predicted']}`.", "",
              "MobileNet is faster here and has fewer parameters than ResNet18, but more",
              "parameters than the Custom CNN. Its sensitivity remains low; the table does",
              "not establish overall superiority or statistical significance. This is one",
              "seed and one reused test partition, with no verified patient or centre",
              "independence and no clinical validation.", "",
              "The full-precision report includes source hashes, the evaluation receipt,",
              "original metric checks and all measured latency samples. Predictions and",
              "checkpoints stay outside Git. Dataset-derived visualizations follow the",
              "attribution and ShareAlike terms in `data/README.md`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-report", type=Path, help="Render saved aggregates without external artifacts")
    parser.add_argument("--json-output", type=Path, default=Path("reports/mobilenet_exploratory_results.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/mobilenet_exploratory_summary.md"))
    args = parser.parse_args()
    if args.output.exists() or (args.from_report is None and args.json_output.exists()):
        raise FileExistsError("Use a new output path; comparison records are preserved")
    report = json.loads(args.from_report.read_text()) if args.from_report else create_comparison_report()
    if args.from_report is None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(render_comparison(report))
    print(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
