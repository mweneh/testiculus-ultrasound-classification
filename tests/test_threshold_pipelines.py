import csv
import json
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from src import evaluation, inference
from src.threshold_config import DEFAULT_THRESHOLD_CONFIG, load_selected_threshold
from src.threshold_selection import create_threshold_configuration
from src.training import CLASS_TO_LABEL


class ConstantProbabilityModel(torch.nn.Module):
    """Produce fixed scores while checking the pipeline's preprocessing."""

    def __init__(self, probability, channels):
        super().__init__()
        self.logits = torch.log(torch.tensor(
            [1 - probability, probability], dtype=torch.float64
        ))
        self.channels = channels

    def forward(self, images):
        assert images.shape[1:] == (self.channels, 224, 224)
        assert not self.training
        assert not torch.is_grad_enabled()
        return self.logits.expand(images.shape[0], -1)


@pytest.fixture
def synthetic_images(tmp_path):
    root = tmp_path / "images"
    for name in CLASS_TO_LABEL:
        folder = root / name
        folder.mkdir(parents=True)
        Image.new("L", (16, 16), color=80).save(folder / "a.png")
    manifest = tmp_path / "validation.csv"
    manifest.write_text(
        "image_path,label\nhomogeneous/a.png,0\ninhomogeneous/a.png,1\n",
        encoding="utf-8",
    )
    return root, manifest


def _mock_checkpoint(monkeypatch, module, model_type, probability=0.4):
    model = ConstantProbabilityModel(probability, 1 if model_type == "custom_cnn" else 3)
    model.eval()
    metadata = {"model_type": model_type, "class_to_idx": dict(CLASS_TO_LABEL)}
    monkeypatch.setattr(module, "load_checkpoint", lambda *args, **kwargs: (model, metadata))
    return metadata


@pytest.mark.parametrize("model_type,expected", [("custom_cnn", 0), ("resnet18", 1)])
def test_evaluation_and_inference_use_same_configured_decision(
    monkeypatch, tmp_path, synthetic_images, model_type, expected,
):
    root, manifest = synthetic_images
    _mock_checkpoint(monkeypatch, evaluation, model_type)
    _mock_checkpoint(monkeypatch, inference, model_type)
    # Curve output is covered separately; exercise real CSV/JSON serialization.
    monkeypatch.setattr(evaluation, "save_roc_pr_plots", lambda *args: None)
    output = tmp_path / "evaluation"
    result = evaluation.evaluate_checkpoint(
        tmp_path / "mock.pth", root, manifest, output, device="cpu"
    )
    predictor = inference.CheckpointPredictor(tmp_path / "mock.pth", device="cpu")
    prediction = predictor.predict(Image.new("RGB", (16, 16), color="gray"))
    threshold = load_selected_threshold(model_type)
    assert result["metrics"]["threshold"] == prediction["threshold"] == threshold
    assert result["threshold_source"] == prediction["threshold_source"] == "validation_configuration"
    assert prediction["prediction"] == expected
    assert prediction["class_name"] == ("inhomogeneous" if expected else "homogeneous")
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(float(row["threshold"]) == threshold for row in rows)
    assert all(int(row["prediction"]) == expected for row in rows)
    assert json.loads((output / "metrics.json").read_text()) == result


def test_alternate_config_and_inclusive_threshold_boundary(monkeypatch, tmp_path, synthetic_images):
    root, manifest = synthetic_images
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "image_path,label,probability_class_1\n"
        "homogeneous/a.png,0,0.1\ninhomogeneous/a.png,1,0.5\n",
        encoding="utf-8",
    )
    config = tmp_path / "alternate.json"
    create_threshold_configuration(
        {"custom_cnn": predictions, "resnet18": predictions}, manifest, config
    )
    for module in (evaluation, inference):
        _mock_checkpoint(monkeypatch, module, "custom_cnn", probability=0.5)
    monkeypatch.setattr(evaluation, "save_roc_pr_plots", lambda *args: None)
    predictor = inference.CheckpointPredictor(
        tmp_path / "mock.pth", threshold_config=config, device="cpu"
    )
    assert predictor.predict(Image.new("L", (16, 16)))["prediction"] == 1
    result = evaluation.evaluate_checkpoint(
        tmp_path / "mock.pth", root, manifest, tmp_path / "out",
        device="cpu", threshold_config=config,
    )
    assert result["metrics"]["threshold"] == 0.5
    assert result["metrics"]["confusion_matrix"]["fp"] == 1
    assert result["metrics"]["confusion_matrix"]["tp"] == 1


def test_explicit_baseline_evaluation_does_not_load_config(monkeypatch, tmp_path, synthetic_images):
    root, manifest = synthetic_images
    _mock_checkpoint(monkeypatch, evaluation, "resnet18")
    monkeypatch.setattr(evaluation, "save_roc_pr_plots", lambda *args: None)
    result = evaluation.evaluate_checkpoint(
        tmp_path / "mock.pth", root, manifest, tmp_path / "out", threshold=0.5, device="cpu"
    )
    assert result["metrics"]["threshold"] == 0.5
    assert result["metrics"]["confusion_matrix"]["tn"] == 1
    assert result["threshold_source"] == "explicit_override"
    assert result["threshold_config"] is None


@pytest.mark.parametrize("module", [evaluation, inference])
def test_missing_configuration_fails_before_opening_images(monkeypatch, tmp_path, module):
    _mock_checkpoint(monkeypatch, module, "custom_cnn")
    with pytest.raises(FileNotFoundError, match="missing.json"):
        if module is evaluation:
            evaluation.evaluate_checkpoint(
                tmp_path / "mock.pth", tmp_path / "no-images", tmp_path / "no-manifest.csv",
                tmp_path / "out", threshold_config=tmp_path / "missing.json", device="cpu",
            )
        else:
            inference.CheckpointPredictor(
                tmp_path / "mock.pth", threshold_config=tmp_path / "missing.json", device="cpu"
            )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("module", [evaluation, inference])
def test_rejects_inverted_checkpoint_class_mapping(monkeypatch, tmp_path, module):
    metadata = _mock_checkpoint(monkeypatch, module, "custom_cnn")
    metadata["class_to_idx"] = {"homogeneous": 1, "inhomogeneous": 0}
    with pytest.raises(ValueError, match="class mapping"):
        if module is evaluation:
            evaluation.evaluate_checkpoint(
                tmp_path / "mock.pth", tmp_path, tmp_path / "unopened.csv", tmp_path / "out"
            )
        else:
            inference.CheckpointPredictor(tmp_path / "mock.pth")


def test_predictor_loads_actual_untrained_checkpoint_without_training(tmp_path):
    from src.models.architectures import CustomCNN
    model = CustomCNN(num_classes=2)
    checkpoint = tmp_path / "untrained.pth"
    torch.save({
        "model_type": "custom_cnn", "architecture": "CustomCNN",
        "class_to_idx": CLASS_TO_LABEL, "model_state_dict": model.state_dict(),
    }, checkpoint)
    predictor = inference.CheckpointPredictor(checkpoint, device="cpu")
    result = predictor.predict(Image.new("L", (16, 16), color=80))
    assert result["threshold"] == load_selected_threshold("custom_cnn")
    assert np.isfinite(result["probability_class_1"])
    assert result["prediction"] == int(result["probability_class_1"] >= result["threshold"])


@pytest.mark.parametrize("config_args", [[], ["--threshold-config", str(DEFAULT_THRESHOLD_CONFIG)]])
def test_evaluation_cli_passes_configuration_to_pipeline(monkeypatch, tmp_path, capsys, config_args):
    captured = {}
    def fake_evaluate(*args, **kwargs):
        captured.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr(evaluation, "evaluate_checkpoint", fake_evaluate)
    monkeypatch.setattr(sys, "argv", [
        "evaluation", "--checkpoint", str(tmp_path / "mock.pth"),
        "--manifest", str(tmp_path / "synthetic.csv"), "--output-dir", str(tmp_path / "out"),
        *config_args,
    ])
    evaluation.main()
    assert captured["threshold"] is None
    assert captured["threshold_config"] == (DEFAULT_THRESHOLD_CONFIG if config_args else None)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_evaluation_cli_rejects_ambiguous_threshold_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "evaluation", "--checkpoint", "unused.pth", "--manifest", "unused.csv",
        "--output-dir", "unused", "--threshold", "0.5", "--threshold-config", "unused.json",
    ])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2
