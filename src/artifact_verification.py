"""Verify downloaded frozen Colab artifacts without training or test-image access."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.evaluation import calculate_metrics
from src.experiment_summary import load_experiment_results, verify_metrics_artifact
from src.gradcam import compute_gradcam, get_target_layer
from src.inference import CheckpointPredictor
from src.threshold_config import DEFAULT_THRESHOLD_CONFIG, LEGACY_MODEL_NAMES as MODEL_NAMES
from src.threshold_selection import verify_threshold_configuration
from src.training import _preprocessing_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_prediction_artifact(predictions_path: Path, metrics_path: Path,
                               threshold: float, *, allow_baseline_mismatch: bool = False) -> dict:
    """Recompute recorded metrics at one fixed threshold, never search candidates."""
    artifact = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    rows = read_rows(predictions_path)
    if len(rows) != artifact["n_samples"] or not rows:
        raise ValueError("Prediction sample count disagrees with original metrics")
    identities, labels, probabilities = {}, [], []
    for row in rows:
        path = Path(row["image_path"])
        label, prediction = int(row["label"]), int(row["prediction"])
        probability, negative = float(row["probability_class_1"]), float(row["probability_class_0"])
        if (label not in (0, 1) or path.is_absolute() or ".." in path.parts
                or len(path.parts) != 2
                or path.parts[0] != ("homogeneous" if label == 0 else "inhomogeneous")):
            raise ValueError("Invalid class-qualified prediction path or label")
        key = path.as_posix()
        if key in identities:
            raise ValueError("Duplicate prediction image path")
        if not all(math.isfinite(p) and 0 <= p <= 1 for p in (probability, negative)):
            raise ValueError("Invalid prediction probability")
        if not math.isclose(probability + negative, 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Class probabilities do not sum to one")
        if float(row["threshold"]) != threshold or prediction != int(probability >= threshold):
            raise ValueError("Stored prediction or threshold disagrees with the fixed decision rule")
        identities[key] = label
        labels.append(label)
        probabilities.append(probability)
    recomputed = calculate_metrics(labels, probabilities, threshold=threshold)
    original = artifact["metrics"]
    if set(original) != set(recomputed):
        raise ValueError("Original and recomputed metric fields disagree")
    discrepancies = {}
    for key, actual in recomputed.items():
        expected = original[key]
        if isinstance(actual, float):
            matches = (type(expected) in (int, float) and math.isfinite(expected)
                       and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
        else:
            matches = actual == expected
        if not matches:
            if key == "majority_class_baseline_accuracy" and allow_baseline_mismatch:
                discrepancies[key] = {"recorded": expected, "recomputed_from_partition": actual}
                continue
            raise ValueError(f"Recomputed {key} disagrees with original metrics")
    identity_bytes = json.dumps(sorted(identities.items()), separators=(",", ":")).encode()
    return {
        "predictions_sha256": sha256(predictions_path),
        "metrics_sha256": sha256(metrics_path),
        "n_samples": len(rows), "class_counts": {str(i): labels.count(i) for i in (0, 1)},
        "sample_paths_and_labels_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "all_metrics_reproduced": not discrepancies,
        "model_metrics_reproduced": True, "discrepancies": discrepancies, "metrics": original,
    }


def verify_downloaded_artifacts(root: Path, data_dir: Path | None = None,
                                check_api: bool = False) -> dict:
    """Check artifacts, trained weights and Grad-CAM; optionally sample validation images.

    The tracked threshold report and user-supplied rounded test summary are
    preserved. No test manifest/images are opened and no optimizer is created.
    """
    root = Path(root)
    results, validation = load_experiment_results()
    downloaded = root / "results/validation_thresholds.json"
    if downloaded.read_bytes() != DEFAULT_THRESHOLD_CONFIG.read_bytes():
        raise ValueError("Downloaded threshold record is not byte-identical to the frozen record")
    manifest = DEFAULT_THRESHOLD_CONFIG.parents[1] / "data/splits/validation.csv"
    validation_sources = {
        name: root / "results" / f"{name}_validation/predictions.csv" for name in MODEL_NAMES
    }
    validation_check = verify_threshold_configuration(DEFAULT_THRESHOLD_CONFIG, manifest, validation_sources)
    report = {
        "format_version": 1,
        "verified": True,
        "threshold_configuration_sha256": sha256(DEFAULT_THRESHOLD_CONFIG),
        "user_supplied_summary_sha256": sha256(DEFAULT_THRESHOLD_CONFIG.with_name("final_test_results.json")),
        "downloaded_threshold_record_byte_identical": True,
        "validation_selection": validation_check,
        "safeguards": {"training_performed": False, "thresholds_changed": False,
                       "test_images_accessed": False, "test_manifest_accessed": False,
                       "test_model_evaluation_rerun": False},
        "verification_environment": {name: version(name) for name in (
            "torch", "torchvision", "numpy", "pandas", "Pillow", "scikit-learn", "fastapi",
        )},
        "concerns": [],
        "models": {},
    }
    predictors, samples = {}, {}
    for name in MODEL_NAMES:
        metrics_path = root / "results" / f"{name}_test/metrics.json"
        summary_check = verify_metrics_artifact(results, name, metrics_path)
        test_check = verify_prediction_artifact(
            root / "results" / f"{name}_test/predictions.csv", metrics_path,
            validation["models"][name]["selected_threshold"],
        )
        if test_check["class_counts"] != {"0": 104, "1": 25}:
            raise ValueError(f"{name}: unexpected test class counts")
        validation_metrics = root / "results" / f"{name}_validation/metrics.json"
        validation_predictions = verify_prediction_artifact(
            validation_sources[name], validation_metrics, 0.5, allow_baseline_mismatch=True
        )
        if validation_predictions["discrepancies"]:
            report["concerns"].append({
                "model_type": name, "artifact": f"results/{name}_validation/metrics.json",
                "issue": "Original validation majority baseline differs from the evaluated partition; model metrics and frozen threshold selection are unaffected. Original file preserved.",
                "discrepancies": validation_predictions["discrepancies"],
            })
        checkpoint_path = root / "training_runs" / name / "best_model.pth"
        checkpoint_hash = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
        if metadata["model_type"] != name or metadata["preprocessing"] != _preprocessing_metadata(name):
            raise ValueError(f"{name}: checkpoint model/preprocessing metadata disagrees with serving code")
        history_path = checkpoint_path.with_name("history.csv")
        history = read_rows(history_path)
        selected_rows = [row for row in history if int(row["epoch"]) == metadata["best_epoch"]]
        if len(selected_rows) != 1 or not math.isclose(
            float(selected_rows[0]["validation_loss"]), metadata["best_validation_loss"],
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(f"{name}: checkpoint epoch/loss disagrees with training history")
        predictor = CheckpointPredictor(checkpoint_path, device="cpu")
        predictors[name] = predictor
        image = Image.fromarray(np.arange(40 * 30, dtype=np.uint8).reshape(30, 40))
        smoke_prediction = predictor.predict(image)
        layer_name, layer = get_target_layer(predictor.model, name)
        inputs = predictor.transform(image).unsqueeze(0)
        heatmap = compute_gradcam(predictor.model, inputs, layer, smoke_prediction["prediction"])
        if (heatmap.shape != (224, 224) or not np.isfinite(heatmap).all()
                or heatmap.min() < 0 or heatmap.max() > 1):
            raise ValueError(f"{name}: invalid trained-model Grad-CAM")
        if predictor.predict(image) != smoke_prediction or any(p.grad is not None for p in predictor.model.parameters()):
            raise ValueError(f"{name}: Grad-CAM altered prediction or parameter gradients")
        sample_check = {"checked": False}
        if data_dir is not None:
            rows = read_rows(validation_sources[name])
            chosen = [next(row for row in rows if int(row["label"]) == label) for label in (0, 1)]
            errors, samples[name] = [], []
            for row in chosen:
                path = Path(data_dir) / row["image_path"]
                with Image.open(path) as sample:
                    prediction = predictor.predict(sample)
                difference = abs(prediction["probability_class_1"] - float(row["probability_class_1"]))
                if difference > 1e-5:
                    raise ValueError(f"{name}: sampled validation probability differs from Colab by {difference}")
                expected_label = int(float(row["probability_class_1"]) >= predictor.threshold)
                if prediction["prediction"] != expected_label:
                    raise ValueError(f"{name}: sampled validation threshold decision differs from Colab")
                errors.append(difference)
                samples[name].append((path, prediction))
            sample_check = {"checked": True, "n_samples": 2, "classes_sampled": [0, 1],
                            "probability_absolute_tolerance": 1e-5,
                            "max_absolute_probability_difference": max(errors),
                            "frozen_threshold_decisions_match": True}
        report["models"][name] = {
            "supplied_test_summary": summary_check, "test_predictions": test_check,
            "validation_predictions": validation_predictions,
            "checkpoint_sha256": checkpoint_hash, "checkpoint_metadata": metadata,
            "history_sha256": sha256(history_path), "epochs_completed": len(history),
            "checkpoint_loads": True, "preprocessing_metadata_matches": True,
            "sampled_validation_inference": sample_check,
            "gradcam": {"verified": True, "input": "synthetic grayscale pattern",
                        "target_layer": layer_name, "heatmap_shape": list(heatmap.shape),
                        "finite_normalized": True, "prediction_unchanged": True,
                        "parameter_gradients_unchanged": True, "constant_map": bool(np.ptp(heatmap) == 0)},
        }
        if sha256(checkpoint_path) != checkpoint_hash:
            raise ValueError(f"{name}: checkpoint file changed during verification")
    hashes = {m["test_predictions"]["sample_paths_and_labels_sha256"] for m in report["models"].values()}
    if len(hashes) != 1:
        raise ValueError("Models were not evaluated on identical test paths/labels")
    report["identical_test_sample_paths_and_labels"] = True
    report["api"] = {"checked": False}
    if check_api:
        from io import BytesIO
        from fastapi.testclient import TestClient
        from src.api import create_app

        paths = {name: root / "training_runs" / name / "best_model.pth" for name in MODEL_NAMES}
        with TestClient(create_app(paths, device="cpu")) as client:
            if client.get("/health").status_code != 200:
                raise ValueError("API readiness check failed")
            for name, predictor in predictors.items():
                checks = samples.get(name, [])
                if not checks:
                    stream = BytesIO()
                    image.save(stream, format="PNG")
                    checks = [(stream.getvalue(), predictor.predict(image))]
                for path, expected in checks:
                    content = path if isinstance(path, bytes) else path.read_bytes()
                    response = client.post(f"/predict?model_type={name}", files={"file": ("input.png", content)})
                    response.raise_for_status()
                    actual = response.json()
                    if any(actual[key] != expected[key] for key in (
                        "prediction", "class_name", "probability_class_1", "threshold",
                    )):
                        raise ValueError(f"{name}: API and CLI predictions disagree")
        report["api"] = {"checked": True, "both_trained_models_load": True,
                         "api_cli_probability_and_decision_agreement": True,
                         "input_partition": "validation" if data_dir is not None else "synthetic"}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, help="Sample two verified validation images per model only")
    parser.add_argument("--check-api", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/artifact_verification.json"))
    args = parser.parse_args()
    report = verify_downloaded_artifacts(args.artifact_root, args.data_dir, args.check_api)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "report": str(args.output), "api": report["api"]}, indent=2))


if __name__ == "__main__":
    main()
