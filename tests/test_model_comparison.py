import json
from pathlib import Path

from src.model_comparison import render_comparison


ROOT = Path(__file__).resolve().parents[1]


def test_saved_exploratory_comparison_preserves_original_metrics():
    report = json.loads((ROOT / "reports/mobilenet_exploratory_results.json").read_text())
    original = json.loads((ROOT / "reports/final_test_results.json").read_text())
    assert report["experiment_status"] == "post-hoc exploratory"
    assert report["original_test_outcomes_previously_observed"] is True
    assert report["original_artifacts_unchanged"] and report["identical_test_paths_and_labels"]
    for name, saved in original["models"].items():
        metrics = report["models"][name]["test"]["metrics"]
        assert metrics["threshold"] == saved["threshold"]
        assert metrics["confusion_matrix"]["rows_actual_columns_predicted"] == saved["confusion_matrix"]["rows_actual_columns_predicted"]
        for key, value in saved["metrics"].items():
            assert round(metrics[key], 4) == value
    assert report["evaluation_receipt"]["evaluation_attempt"] == 1
    assert report["evaluation_receipt"]["complete"]


def test_saved_efficiency_measurements_have_common_conditions_and_frozen_weights():
    report = json.loads((ROOT / "reports/mobilenet_exploratory_results.json").read_text())
    conditions = report["benchmark_conditions"]
    assert conditions["device"] == "cpu" and conditions["batch_size"] == 1
    assert conditions["input_size"] == [224, 224]
    assert conditions["warmup"] >= 1 and conditions["repeats"] >= 2
    identities = set()
    for model in report["models"].values():
        efficiency = model["efficiency"]
        assert efficiency["weight_status"] == "trained checkpoint"
        assert len(efficiency["latency"]["measurements_ms"]) == conditions["repeats"]
        assert efficiency["latency"]["median_ms"] > 0
        assert efficiency["parameters"] > 0 and efficiency["checkpoint_bytes"] > 0
        assert model["test"]["class_counts"] == {"0": 104, "1": 25}
        identities.add(model["test"]["sample_paths_and_labels_sha256"])
    assert len(identities) == 1
    assert report["models"]["mobilenet_v3_small"]["efficiency"]["checkpoint_sha256"] == report["freeze"]["sources"]["checkpoint"]["sha256"]


def test_comparison_summary_reproduces_from_aggregates_without_external_artifacts():
    report = json.loads((ROOT / "reports/mobilenet_exploratory_results.json").read_text())
    assert render_comparison(report) == (ROOT / "reports/mobilenet_exploratory_summary.md").read_text()
    mobile = report["models"]["mobilenet_v3_small"]["test"]["metrics"]
    assert mobile["threshold"] == report["freeze"]["threshold"]
    assert report["freeze"]["test_images_accessed_for_selection"] is False
