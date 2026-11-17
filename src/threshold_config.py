"""Load the frozen validation threshold record without accessing any data."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


DEFAULT_THRESHOLD_CONFIG = (
    Path(__file__).resolve().parents[1] / "reports" / "validation_thresholds.json"
)
MODEL_NAMES = ("custom_cnn", "resnet18")
METRIC_KEYS = ("precision", "recall_sensitivity", "specificity", "f1")
SELECTION_CRITERIA = {
    "partition": "validation",
    "positive_class": 1,
    "objective": "maximize positive-class F1-score",
    "candidate_thresholds": "every unique probability_class_1 score",
    "prediction_rule": "probability_class_1 >= threshold",
    "tie_break": "choose the highest threshold among equal maximum F1 scores",
    "test_partition_accessed": False,
}


def _object(value, required, context: str) -> dict:
    if not isinstance(value, dict) or not set(required).issubset(value):
        raise ValueError(f"{context} must be an object containing {sorted(required)}")
    return value


def _unit_number(value, context: str) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{context} must be a finite number in [0, 1]")
    return float(value)


def _text(value, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")


def _hash(value, context: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")


def _metrics(value, keys, context: str) -> None:
    _object(value, keys, context)
    for key in keys:
        if value[key] is not None:
            _unit_number(value[key], f"{context}.{key}")


def validate_threshold_configuration(configuration: dict) -> None:
    """Check version-1 provenance, model records, sweeps, and selection rule.

    This checks the recorded evidence for consistency. Source prediction hashes
    and recomputed metrics require the separate validation-only verifier.
    """
    _object(configuration, (
        "format_version", "validation_manifest", "validation_manifest_sha256",
        "selection_criteria", "models",
    ), "Threshold configuration")
    if type(configuration["format_version"]) is not int or configuration["format_version"] != 1:
        raise ValueError("Unsupported threshold configuration format_version")
    _text(configuration["validation_manifest"], "validation_manifest")
    _hash(configuration["validation_manifest_sha256"], "validation_manifest_sha256")
    criteria = _object(configuration["selection_criteria"], SELECTION_CRITERIA, "selection_criteria")
    for key, expected in SELECTION_CRITERIA.items():
        if type(criteria[key]) is not type(expected) or criteria[key] != expected:
            raise ValueError(f"Unsupported selection_criteria.{key}: {criteria[key]!r}")

    models = _object(configuration["models"], MODEL_NAMES, "models")
    if set(models) != set(MODEL_NAMES):
        raise ValueError(f"models must contain exactly {MODEL_NAMES}")
    for name, model in models.items():
        _object(model, (
            "predictions_file", "predictions_sha256", "n_validation_samples",
            "majority_class_baseline_accuracy", "selected_threshold",
            "metrics_at_0.5", "metrics_at_selected_threshold", "threshold_sweep",
        ), name)
        _text(model["predictions_file"], f"{name}.predictions_file")
        _hash(model["predictions_sha256"], f"{name}.predictions_sha256")
        n = model["n_validation_samples"]
        if type(n) is not int or n < 1:
            raise ValueError(f"{name}.n_validation_samples must be a positive integer")
        baseline = _unit_number(model["majority_class_baseline_accuracy"], f"{name}.baseline")
        if baseline < 0.5:
            raise ValueError(f"{name}.baseline cannot be below 0.5 for binary labels")
        selected = _unit_number(model["selected_threshold"], f"{name}.selected_threshold")
        for key in ("metrics_at_0.5", "metrics_at_selected_threshold"):
            _metrics(model[key], METRIC_KEYS, f"{name}.{key}")

        sweep = model["threshold_sweep"]
        if not isinstance(sweep, list) or not 1 <= len(sweep) <= n:
            raise ValueError(f"{name}.threshold_sweep must contain 1 to {n} candidates")
        previous = -1.0
        for row in sweep:
            keys = ("precision", "recall", "specificity", "f1")
            _object(row, ("threshold", *keys), f"{name}.threshold_sweep row")
            threshold = _unit_number(row["threshold"], f"{name}.sweep threshold")
            if threshold <= previous:
                raise ValueError(f"{name}.threshold_sweep thresholds must be strictly increasing")
            previous = threshold
            _metrics(row, keys, f"{name}.threshold_sweep row")
            # Every candidate predicts at least one positive; selection requires
            # positive validation examples, so only specificity can be undefined.
            for key in ("precision", "recall", "f1"):
                _unit_number(row[key], f"{name}.threshold_sweep.{key}")
        best = max(sweep, key=lambda row: (row["f1"], row["threshold"]))
        if selected != best["threshold"]:
            raise ValueError(f"{name}.selected_threshold disagrees with maximum F1 / highest-threshold tie-break")
        for key in METRIC_KEYS:
            sweep_key = "recall" if key == "recall_sensitivity" else key
            if model["metrics_at_selected_threshold"][key] != best[sweep_key]:
                raise ValueError(f"{name}.metrics_at_selected_threshold.{key} disagrees with sweep")


def _unique_keys(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def load_threshold_configuration(path: Path = DEFAULT_THRESHOLD_CONFIG) -> dict:
    """Return the complete validated record; never follow its artifact paths."""
    with Path(path).open(encoding="utf-8") as stream:
        configuration = json.load(
            stream, object_pairs_hook=_unique_keys, parse_constant=_reject_constant
        )
    validate_threshold_configuration(configuration)
    return configuration


def load_selected_threshold(model_type: str,
                            path: Path = DEFAULT_THRESHOLD_CONFIG) -> float:
    """Look up a threshold by checkpoint model_type; fail instead of falling back."""
    if model_type not in MODEL_NAMES:
        raise ValueError(f"Unsupported threshold model_type: {model_type}")
    configuration = load_threshold_configuration(path)
    return float(configuration["models"][model_type]["selected_threshold"])


def resolve_threshold(model_type: str, threshold: float | None = None,
                      threshold_config: Path | None = None) -> float:
    """Use the frozen record by default, or one explicit numeric override."""
    if model_type not in MODEL_NAMES:
        raise ValueError(f"Unsupported threshold model_type: {model_type}")
    if threshold is not None:
        if threshold_config is not None:
            raise ValueError("Provide threshold or threshold_config, not both")
        return _unit_number(threshold, "threshold")
    path = DEFAULT_THRESHOLD_CONFIG if threshold_config is None else threshold_config
    return load_selected_threshold(model_type, path)
