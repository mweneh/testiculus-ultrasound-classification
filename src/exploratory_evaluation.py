"""Freeze validation evidence before a single post-hoc MobileNet test evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from src.evaluation import calculate_metrics, evaluate_checkpoint
from src.threshold_config import load_threshold_configuration
from src.threshold_selection import load_validation_predictions, verify_threshold_configuration
from src.training import CLASS_TO_LABEL, load_checkpoint


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_original_artifacts(protocol: dict) -> None:
    for name, expected in protocol["original_artifact_sha256"].items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Original frozen artifact changed: {name}")


def freeze_experiment(*, checkpoint, threshold_config, validation_predictions,
                      validation_manifest, training_summary, protocol_path, output) -> dict:
    """Write aggregate evidence and hashes without opening any test image/manifest."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite the frozen experiment")
    protocol = json.loads(Path(protocol_path).read_text())
    if protocol.get("original_test_outcomes_previously_observed") is not True:
        raise ValueError("The extension must be disclosed as post-hoc exploratory")
    check_original_artifacts(protocol)
    configuration = load_threshold_configuration(threshold_config)
    if set(configuration["models"]) != {"mobilenet_v3_small"}:
        raise ValueError("Expected a separate MobileNet validation configuration")
    verification = verify_threshold_configuration(threshold_config, validation_manifest,
                                                  {"mobilenet_v3_small": validation_predictions})
    model, metadata = load_checkpoint(checkpoint, device="cpu")
    summary = json.loads(Path(training_summary).read_text())
    if (metadata["model_type"] != "mobilenet_v3_small" or metadata["class_to_idx"] != CLASS_TO_LABEL
            or metadata["best_epoch"] != summary["best_epoch"]
            or metadata["best_validation_loss"] != summary["best_validation_loss"]):
        raise ValueError("Checkpoint and training summary disagree")
    if summary["manifest_sha256"] != protocol["manifest_sha256"]:
        raise ValueError("Training used different manifests than the declared protocol")
    del model
    labels, probabilities = load_validation_predictions(validation_predictions, validation_manifest)
    threshold = configuration["models"]["mobilenet_v3_small"]["selected_threshold"]
    files = {"checkpoint": checkpoint, "threshold_configuration": threshold_config,
             "validation_predictions": validation_predictions, "validation_manifest": validation_manifest,
             "training_summary": training_summary, "protocol": protocol_path}
    frozen = {
        "format_version": 1, "model_type": "mobilenet_v3_small",
        "experiment_status": "post-hoc exploratory",
        "original_test_outcomes_previously_observed": True,
        "frozen_before_test_evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {key: {"path": str(path), "sha256": sha256(path)} for key, path in files.items()},
        "threshold": threshold, "threshold_verification": verification,
        "validation_metrics_at_0.5": calculate_metrics(labels, probabilities, threshold=.5),
        "validation_metrics_at_frozen_threshold": calculate_metrics(labels, probabilities, threshold=threshold),
        "training": summary,
        "fine_tuning_decision": "No additional fine-tuning: retain the declared frozen-backbone experiment. Validation evidence does not establish clear underfitting requiring an additional run.",
        "evaluation_batch_size": 1, "evaluation_device": "cpu",
        "test_images_accessed_for_selection": False,
        "original_artifact_hashes_verified": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    return frozen


def evaluate_once(freeze_path: Path, data_dir: Path, test_manifest: Path, output_dir: Path) -> dict:
    """Reserve the output directory before accessing test data; refuse every rerun."""
    frozen = json.loads(Path(freeze_path).read_text())
    if (frozen.get("model_type") != "mobilenet_v3_small"
            or frozen.get("experiment_status") != "post-hoc exploratory"
            or frozen.get("original_test_outcomes_previously_observed") is not True):
        raise ValueError("Require a frozen post-hoc MobileNet experiment")
    for record in frozen["sources"].values():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError("Frozen source hash changed; test evaluation refused")
    config_path = Path(frozen["sources"]["threshold_configuration"]["path"])
    configuration = load_threshold_configuration(config_path)
    if configuration["models"]["mobilenet_v3_small"]["selected_threshold"] != frozen["threshold"]:
        raise ValueError("Frozen threshold changed")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()  # Even a failed/partial evaluation cannot be repeated automatically.
    receipt = {
        "experiment_status": "post-hoc exploratory", "evaluation_attempt": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": sha256(freeze_path), "complete": False,
        "test_manifest": str(test_manifest), "threshold": frozen["threshold"],
    }
    receipt_path = output_dir / "evaluation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    result = evaluate_checkpoint(
        Path(frozen["sources"]["checkpoint"]["path"]), data_dir, test_manifest, output_dir,
        threshold_config=config_path, device="cpu", batch_size=1, num_workers=0,
    )
    receipt.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                   test_manifest_sha256=sha256(test_manifest),
                   metrics_sha256=sha256(output_dir / "metrics.json"),
                   predictions_sha256=sha256(output_dir / "predictions.csv"))
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--checkpoint", type=Path, default=Path("training_runs/mobilenet_v3_small/best_model.pth"))
    freeze.add_argument("--threshold-config", type=Path, default=Path("reports/mobilenet_v3_small_thresholds.json"))
    freeze.add_argument("--validation-predictions", type=Path, default=Path("results/mobilenet_v3_small_validation/predictions.csv"))
    freeze.add_argument("--validation-manifest", type=Path, default=Path("data/splits/validation.csv"))
    freeze.add_argument("--training-summary", type=Path, default=Path("training_runs/mobilenet_v3_small/training_summary.json"))
    freeze.add_argument("--protocol-path", type=Path, default=Path("reports/mobilenet_exploratory_protocol.json"))
    freeze.add_argument("--output", type=Path, default=Path("reports/mobilenet_v3_small_freeze.json"))
    evaluate = subparsers.add_parser("evaluate-once")
    evaluate.add_argument("--freeze", type=Path, default=Path("reports/mobilenet_v3_small_freeze.json"))
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    evaluate.add_argument("--test-manifest", type=Path, default=Path("data/splits/test.csv"))
    evaluate.add_argument("--output-dir", type=Path, default=Path("results/mobilenet_v3_small_test"))
    args = parser.parse_args()
    if args.command == "freeze":
        kwargs = vars(args).copy()
        kwargs.pop("command")
        result = freeze_experiment(**kwargs)
    else:
        result = evaluate_once(args.freeze, args.data_dir, args.test_manifest, args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
