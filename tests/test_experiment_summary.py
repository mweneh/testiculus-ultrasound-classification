import copy
import json

import pytest

from src.experiment_summary import (
    DEFAULT_TEST_RESULTS,
    DEFAULT_VERIFICATION,
    load_experiment_results,
    load_verification_report,
    render_summary,
    verify_metrics_artifact,
)


def test_frozen_results_and_validation_are_distinct_and_summary_is_reproducible():
    results, validation = load_experiment_results()
    assert results["provenance"]["original_artifacts_verified"] is False
    assert results["n_samples"] == 129
    assert results["models"]["custom_cnn"]["metrics"]["f1"] == 0.2857
    assert results["models"]["resnet18"]["metrics"]["roc_auc"] == 0.7473
    assert validation["models"]["custom_cnn"]["metrics_at_selected_threshold"]["f1"] != 0.2857
    assert render_summary(results, validation, load_verification_report()) == DEFAULT_TEST_RESULTS.with_name("experiment_summary.md").read_text()
    assert "remain unverified" in render_summary(results, validation)


@pytest.mark.parametrize("change", [
    lambda r: r.update(partition="validation"),
    lambda r: r.update(n_samples=130),
    lambda r: r.update(majority_class_baseline_accuracy=0.9),
    lambda r: r["models"]["custom_cnn"].update(threshold=0.5),
    lambda r: r["models"]["resnet18"]["metrics"].update(accuracy=0.9),
    lambda r: r["models"]["custom_cnn"]["metrics"].update(roc_auc=None),
    lambda r: r["models"]["custom_cnn"]["confusion_matrix"].update(rows_actual_columns_predicted=[[70, 35], [15, 10]]),
])
def test_summary_detects_inconsistent_frozen_counts_metrics_or_thresholds(tmp_path, change):
    results = json.loads(DEFAULT_TEST_RESULTS.read_text())
    change(results)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    with pytest.raises(ValueError):
        load_experiment_results(path)


def test_cross_checks_original_metrics_format_without_changing_record(tmp_path):
    results, _ = load_experiment_results()
    original = copy.deepcopy(results)
    record = results["models"]["custom_cnn"]
    artifact = {
        "model_type": "custom_cnn", "n_samples": 129,
        "metrics": {**record["metrics"], "positive_class": 1,
                    "threshold": record["threshold"],
                    "confusion_matrix": record["confusion_matrix"]},
    }
    artifact["metrics"]["accuracy"] = 79 / 129
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(artifact))
    verified = verify_metrics_artifact(results, "custom_cnn", path)
    assert verified["verified"] and len(verified["sha256"]) == 64
    assert results == original
    artifact["metrics"]["roc_auc"] = 0.9
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="roc_auc"):
        verify_metrics_artifact(results, "custom_cnn", path)


def test_verified_summary_reports_original_artifacts_and_legacy_baseline():
    results, validation = load_experiment_results()
    verification = load_verification_report()
    assert verification["api"]["checked"]
    assert verification["identical_test_sample_paths_and_labels"]
    assert len(verification["concerns"]) == 2
    summary = render_summary(results, validation, verification)
    assert "All full-precision test metrics" in summary
    assert "0.8093023" in summary and "0.8062016" in summary
    assert "| custom_cnn | 15 | 8 |" in summary
    assert "| resnet18 | 14 | 7 |" in summary


@pytest.mark.parametrize("change", [
    lambda r: r.update(verified=False),
    lambda r: r.update(threshold_configuration_sha256="0" * 64),
    lambda r: r.update(user_supplied_summary_sha256="0" * 64),
])
def test_rejects_unsuccessful_or_stale_verification_evidence(tmp_path, change):
    report = json.loads(DEFAULT_VERIFICATION.read_text())
    change(report)
    path = tmp_path / "verification.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_verification_report(path)
