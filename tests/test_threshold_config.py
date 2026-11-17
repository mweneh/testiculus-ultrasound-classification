import copy
import hashlib
import json
import pytest

from src.threshold_config import (
    DEFAULT_THRESHOLD_CONFIG,
    load_selected_threshold,
    load_threshold_configuration,
    resolve_threshold,
    validate_threshold_configuration,
)
from src.threshold_selection import (
    create_threshold_configuration,
    threshold_sweep,
    verify_threshold_configuration,
)


@pytest.fixture
def experiment(tmp_path):
    manifest = tmp_path / "validation.csv"
    manifest.write_text(
        "image_path,label\n"
        "homogeneous/a.png,0\nhomogeneous/b.png,0\n"
        "inhomogeneous/a.png,1\ninhomogeneous/b.png,1\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "image_path,label,probability_class_1\n"
        "homogeneous/a.png,0,0.1\nhomogeneous/b.png,0,0.6\n"
        "inhomogeneous/a.png,1,0.4\ninhomogeneous/b.png,1,0.9\n",
        encoding="utf-8",
    )
    files = {"custom_cnn": predictions, "resnet18": predictions}
    output = tmp_path / "thresholds.json"
    configuration = create_threshold_configuration(files, manifest, output)
    return configuration, output, manifest, files


def test_colab_record_is_preserved_and_matches_validation_manifest(monkeypatch, tmp_path):
    raw = DEFAULT_THRESHOLD_CONFIG.read_bytes()
    # Freeze the entire experimental record, including hashes and all metrics.
    assert hashlib.sha256(raw).hexdigest() == (
        "57c39b8495118a91946f54b9c888e1e08b0eaaa9f23ec35d76b668c6022b1349"
    )
    manifest = DEFAULT_THRESHOLD_CONFIG.parents[1] / "data/splits/validation.csv"
    monkeypatch.chdir(tmp_path)
    assert load_selected_threshold("custom_cnn") == 0.4541136622428894
    assert load_selected_threshold("resnet18") == 0.3251684010028839
    result = verify_threshold_configuration(DEFAULT_THRESHOLD_CONFIG, manifest)
    assert result["validation_manifest_sha256_verified"]
    for model in result["models"].values():
        assert model["n_validation_samples"] == 129
        assert model["recorded_sweep_verified"]
        assert not model["source_predictions_verified"]
    assert DEFAULT_THRESHOLD_CONFIG.read_bytes() == raw


@pytest.mark.parametrize("field,value", [
    ("selected_threshold", -0.1),
    ("selected_threshold", 1.1),
    ("selected_threshold", True),
    ("selected_threshold", "0.4"),
    ("selected_threshold", None),
    ("selected_threshold", float("inf")),
    ("selected_threshold", float("nan")),
    ("selected_threshold", 0.5),
    ("predictions_sha256", "invalid"),
    ("predictions_file", ""),
    ("n_validation_samples", 0),
    ("n_validation_samples", True),
    ("majority_class_baseline_accuracy", 0.4),
    ("threshold_sweep", []),
    ("threshold_sweep", {}),
    ("metrics_at_0.5", {}),
])
def test_rejects_invalid_model_records(experiment, field, value):
    configuration = copy.deepcopy(experiment[0])
    configuration["models"]["custom_cnn"][field] = value
    with pytest.raises(ValueError):
        validate_threshold_configuration(configuration)


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(format_version=2),
    lambda c: c.update(format_version=True),
    lambda c: c.update(validation_manifest_sha256="bad"),
    lambda c: c.update(models={}),
    lambda c: c["models"].pop("resnet18"),
    lambda c: c["selection_criteria"].update(partition="test"),
    lambda c: c["selection_criteria"].update(test_partition_accessed=True),
    lambda c: c["selection_criteria"].update(positive_class=True),
    lambda c: c["models"]["custom_cnn"]["threshold_sweep"].reverse(),
    lambda c: c["models"]["custom_cnn"]["threshold_sweep"][1].update(threshold=0.1),
    lambda c: c["models"]["custom_cnn"]["threshold_sweep"][0].update(f1=None),
    lambda c: c["models"]["custom_cnn"]["metrics_at_selected_threshold"].update(f1=0.1),
])
def test_rejects_inconsistent_structure_criteria_and_sweep(experiment, mutation):
    configuration = copy.deepcopy(experiment[0])
    mutation(configuration)
    with pytest.raises(ValueError):
        validate_threshold_configuration(configuration)


@pytest.mark.parametrize("content", [
    '{"format_version": 1, "format_version": 1}',
    '{"format_version": NaN}',
    '{"format_version": Infinity}',
    '{"format_version":',
    '[]',
    'null',
])
def test_loader_rejects_invalid_json(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_threshold_configuration(path)


def test_loader_fails_without_fallback_and_override_is_explicit(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        resolve_threshold("custom_cnn", threshold_config=missing)
    with pytest.raises(ValueError, match="model_type"):
        load_selected_threshold("unknown")
    with pytest.raises(ValueError, match="not both"):
        resolve_threshold("custom_cnn", threshold=0.5, threshold_config=missing)
    assert resolve_threshold("custom_cnn", threshold=0.5) == 0.5


def test_exact_f1_tie_uses_highest_threshold():
    sweep = threshold_sweep([1, 0, 0, 1], [0.2, 0.3, 0.4, 0.5])
    assert sweep[0]["f1"] == sweep[-1]["f1"] == 2 / 3
    assert max(sweep, key=lambda row: (row["f1"], row["threshold"]))["threshold"] == 0.5


def test_verifier_reproduces_relocated_sources_without_modifying_record(experiment, tmp_path):
    _, configuration, manifest, files = experiment
    raw = configuration.read_bytes()
    relocated = tmp_path / "relocated.csv"
    relocated.write_bytes(files["custom_cnn"].read_bytes())
    result = verify_threshold_configuration(
        configuration, manifest, {name: relocated for name in files}
    )
    assert all(model["source_predictions_verified"] for model in result["models"].values())
    assert configuration.read_bytes() == raw


@pytest.mark.parametrize("source", ["manifest", "predictions"])
def test_verifier_rejects_changed_source_hash(experiment, source):
    _, configuration, manifest, files = experiment
    path = manifest if source == "manifest" else files["custom_cnn"]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_threshold_configuration(configuration, manifest, files)


@pytest.mark.parametrize("field", ["metrics_at_0.5", "threshold_sweep"])
def test_verifier_detects_metrics_that_do_not_reproduce(experiment, field):
    record, configuration, manifest, files = experiment
    metrics = record["models"]["custom_cnn"][field]
    if isinstance(metrics, list):
        metrics = metrics[0]
    metrics["precision"] = 0.1
    configuration.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match=f"recomputed {field}"):
        verify_threshold_configuration(configuration, manifest, files)


def test_partial_source_verification_is_reported_per_model(experiment):
    _, configuration, manifest, files = experiment
    result = verify_threshold_configuration(
        configuration, manifest, {"custom_cnn": files["custom_cnn"]}
    )
    assert result["models"]["custom_cnn"]["source_predictions_verified"]
    assert not result["models"]["resnet18"]["source_predictions_verified"]
