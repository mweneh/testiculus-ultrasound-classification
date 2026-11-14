import numpy as np

from src.evaluation import calculate_metrics, save_roc_pr_plots


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
    assert metrics["majority_class_baseline_accuracy"] == 696 / 860
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
