import csv
import json

import pytest

from src.artifact_verification import verify_prediction_artifact
from src.evaluation import calculate_metrics


@pytest.fixture
def saved_predictions(tmp_path):
    predictions, metrics = tmp_path / "predictions.csv", tmp_path / "metrics.json"
    rows = [
        {"image_path": "homogeneous/a.png", "label": "0", "prediction": "0",
         "probability_class_0": "0.8", "probability_class_1": "0.2", "threshold": "0.4"},
        {"image_path": "inhomogeneous/b.png", "label": "1", "prediction": "1",
         "probability_class_0": "0.6", "probability_class_1": "0.4", "threshold": "0.4"},
    ]
    with predictions.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics.write_text(json.dumps({
        "model_type": "custom_cnn", "n_samples": 2,
        "metrics": calculate_metrics([0, 1], [0.2, 0.4], threshold=0.4),
    }))
    return predictions, metrics, rows


def test_saved_predictions_reproduce_metrics_without_modifying_sources(saved_predictions):
    predictions, metrics, _ = saved_predictions
    before = (predictions.read_bytes(), metrics.read_bytes())
    result = verify_prediction_artifact(predictions, metrics, 0.4)
    assert result["all_metrics_reproduced"]
    assert result["class_counts"] == {"0": 1, "1": 1}
    assert result["metrics"]["roc_auc"] == 1
    assert (predictions.read_bytes(), metrics.read_bytes()) == before


@pytest.mark.parametrize("change", [
    lambda r: r[1].update(image_path=r[0]["image_path"], label="0"),
    lambda r: r[0].update(image_path="../private.png"),
    lambda r: r[0].update(image_path="inhomogeneous/a.png"),
    lambda r: r[0].update(probability_class_1="nan"),
    lambda r: r[0].update(probability_class_0="0.4"),
    lambda r: r[1].update(prediction="0"),
    lambda r: r[0].update(threshold="0.5"),
])
def test_rejects_inconsistent_paths_probabilities_and_decisions(saved_predictions, change):
    predictions, metrics, rows = saved_predictions
    change(rows)
    with predictions.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        verify_prediction_artifact(predictions, metrics, 0.4)


def test_detects_incorrect_ranking_metric_in_original_json(saved_predictions):
    predictions, metrics, _ = saved_predictions
    record = json.loads(metrics.read_text())
    record["metrics"]["average_precision"] = 0.5
    metrics.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="average_precision"):
        verify_prediction_artifact(predictions, metrics, 0.4)


def test_legacy_baseline_is_reported_explicitly_without_waiving_other_metrics(saved_predictions):
    predictions, metrics, _ = saved_predictions
    record = json.loads(metrics.read_text())
    record["metrics"]["majority_class_baseline_accuracy"] = 0.8
    metrics.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="majority_class_baseline_accuracy"):
        verify_prediction_artifact(predictions, metrics, 0.4)
    result = verify_prediction_artifact(predictions, metrics, 0.4, allow_baseline_mismatch=True)
    assert not result["all_metrics_reproduced"]
    assert result["model_metrics_reproduced"]
    assert result["discrepancies"] == {
        "majority_class_baseline_accuracy": {"recorded": 0.8, "recomputed_from_partition": 0.5}
    }
    record["metrics"]["roc_auc"] = 0.3
    metrics.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="roc_auc"):
        verify_prediction_artifact(predictions, metrics, 0.4, allow_baseline_mismatch=True)
