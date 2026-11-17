import numpy as np

from src.evaluation import calculate_metrics, save_roc_pr_plots
from src.threshold_selection import (
    create_threshold_configuration,
    load_validation_predictions,
    threshold_sweep,
)


def test_classification_metrics_for_positive_class_one():
    metrics = calculate_metrics(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.4, 0.9],
        threshold=0.5,
    )
    assert metrics["roc_auc"] == 0.75
    assert np.isclose(metrics["average_precision"], 5 / 6)
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall_sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"]["rows_actual_columns_predicted"] == [
        [1, 1], [1, 1]
    ]
    assert metrics["majority_class_baseline_accuracy"] == 0.5
    assert metrics["undefined_metrics"] == []


def test_metrics_mark_undefined_values_explicitly():
    metrics = calculate_metrics(labels=[0, 0], probabilities=[0.1, 0.2])
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["precision"] is None
    assert metrics["recall_sensitivity"] is None
    assert metrics["specificity"] == 1.0
    assert metrics["f1"] is None
    assert set(metrics["undefined_metrics"]) == {
        "roc_auc", "average_precision", "balanced_accuracy", "precision",
        "recall_sensitivity", "f1",
    }


def test_curve_plots_are_saved_for_synthetic_predictions(tmp_path):
    paths = save_roc_pr_plots(
        labels=[0, 0, 1, 1],
        probabilities=[0.1, 0.8, 0.4, 0.9],
        output_dir=tmp_path,
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_majority_baseline_uses_evaluated_label_distribution():
    metrics = calculate_metrics(
        labels=[0] * 104 + [1] * 25,
        probabilities=[0.2] * 104 + [0.8] * 25,
    )
    assert np.isclose(metrics["majority_class_baseline_accuracy"], 104 / 129)


def test_threshold_sweep_selects_maximum_positive_f1():
    labels = [0, 0, 0, 1, 1, 1]
    probabilities = [0.1, 0.4, 0.7, 0.3, 0.6, 0.9]
    sweep = threshold_sweep(labels, probabilities)
    best = max(sweep, key=lambda row: (row["f1"], row["threshold"]))
    assert best["threshold"] == 0.3
    assert best["f1"] == 0.75


def test_validation_predictions_must_match_validation_manifest(tmp_path):
    manifest = tmp_path / "validation.csv"
    predictions = tmp_path / "predictions.csv"
    manifest.write_text(
        "image_path,label\nhomogeneous/a.png,0\ninhomogeneous/b.png,1\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "image_path,label,probability_class_1\n"
        "homogeneous/a.png,0,0.2\ninhomogeneous/b.png,1,0.7\n",
        encoding="utf-8",
    )
    labels, probabilities = load_validation_predictions(predictions, manifest)
    assert labels.tolist() == [0, 1]
    assert probabilities.tolist() == [0.2, 0.7]

    predictions.write_text(
        "image_path,label,probability_class_1\n"
        "homogeneous/a.png,0,0.2\ntest/unknown.png,1,0.7\n",
        encoding="utf-8",
    )
    try:
        load_validation_predictions(predictions, manifest)
    except ValueError as error:
        assert "match the validation manifest exactly" in str(error)
    else:
        raise AssertionError("non-validation predictions were accepted")


def test_threshold_configuration_saves_criteria_and_requested_metrics(tmp_path):
    manifest = tmp_path / "validation.csv"
    manifest.write_text(
        "image_path,label\n"
        "homogeneous/a.png,0\nhomogeneous/b.png,0\n"
        "inhomogeneous/a.png,1\ninhomogeneous/b.png,1\n",
        encoding="utf-8",
    )
    custom = tmp_path / "custom.csv"
    resnet = tmp_path / "resnet.csv"
    csv_content = (
        "image_path,label,probability_class_1\n"
        "homogeneous/a.png,0,0.1\n"
        "homogeneous/b.png,0,0.6\n"
        "inhomogeneous/a.png,1,0.4\n"
        "inhomogeneous/b.png,1,0.9\n"
    )
    custom.write_text(csv_content, encoding="utf-8")
    resnet.write_text(csv_content, encoding="utf-8")
    output = tmp_path / "validation_thresholds.json"

    configuration = create_threshold_configuration(
        {"custom_cnn": custom, "resnet18": resnet}, manifest, output
    )

    assert output.is_file()
    assert configuration["selection_criteria"]["partition"] == "validation"
    assert configuration["selection_criteria"]["test_partition_accessed"] is False
    assert len(configuration["validation_manifest_sha256"]) == 64
    for model in configuration["models"].values():
        assert model["majority_class_baseline_accuracy"] == 0.5
        assert model["selected_threshold"] == 0.4
        assert model["metrics_at_0.5"].keys() == {
            "precision", "recall_sensitivity", "specificity", "f1"
        }
        assert model["metrics_at_selected_threshold"]["f1"] == 0.8
        assert model["predictions_sha256"]
