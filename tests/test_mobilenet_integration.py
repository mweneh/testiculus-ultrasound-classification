from io import BytesIO
import json
import math
import sys

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import torch

from src import api, evaluation, threshold_config, threshold_selection, exploratory_evaluation
from src.gradcam import compute_gradcam, get_target_layer
from src.inference import CheckpointPredictor
from src.models.architectures import build_mobilenet_v3_small
from src.training import _save_checkpoint, load_checkpoint


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "images"
    manifest = tmp_path / "validation.csv"
    predictions = tmp_path / "validation_predictions.csv"
    paths = ["homogeneous/a.png", "inhomogeneous/a.png", "homogeneous/b.png", "inhomogeneous/b.png"]
    labels = [0, 1, 0, 1]
    manifest.write_text("image_path,label\n" + "".join(f"{p},{y}\n" for p, y in zip(paths, labels)))
    predictions.write_text("image_path,label,probability_class_1\n" + "".join(
        f"{p},{y},{probability}\n" for p, y, probability in zip(paths, labels, [.2, .4, .4, .6])))
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (40, 30), color=80).save(root / path)
    config = tmp_path / "thresholds.json"
    threshold_selection.create_threshold_configuration({"mobilenet_v3_small": predictions}, manifest, config)
    model = build_mobilenet_v3_small(pretrained=False)
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([0., math.log(.45 / .55)]))
    checkpoint = tmp_path / "untrained_mobile.pth"
    _save_checkpoint(checkpoint, model, "mobilenet_v3_small", False, 42, 1, .7,
                     torch.ones(2), {"freeze_backbone": True})
    return root, manifest, predictions, config, checkpoint


def test_exploratory_threshold_selection_and_verification(experiment):
    _, manifest, predictions, config, _ = experiment
    record = threshold_config.load_threshold_configuration(config)
    assert record["format_version"] == 2
    assert record["experiment_status"] == "post-hoc exploratory"
    assert record["original_test_outcomes_previously_observed"] is True
    assert threshold_config.load_selected_threshold("mobilenet_v3_small", config) == .4
    result = threshold_selection.verify_threshold_configuration(config, manifest, {"mobilenet_v3_small": predictions})
    assert result["models"]["mobilenet_v3_small"]["source_predictions_verified"]
    record["original_test_outcomes_previously_observed"] = False
    with pytest.raises(ValueError, match="post-hoc"):
        threshold_config.validate_threshold_configuration(record)


def test_model_specific_defaults_preserve_original_thresholds(experiment, monkeypatch):
    _, _, _, config, checkpoint = experiment
    monkeypatch.setattr(threshold_config, "MOBILENET_THRESHOLD_CONFIG", config)
    predictor = CheckpointPredictor(checkpoint, device="cpu")
    assert predictor.threshold_config == config
    assert threshold_config.load_selected_threshold("custom_cnn") == 0.4541136622428894
    assert threshold_config.load_selected_threshold("resnet18") == 0.3251684010028839
    with pytest.raises(ValueError, match="no record"):
        threshold_config.load_selected_threshold("mobilenet_v3_small", threshold_config.DEFAULT_THRESHOLD_CONFIG)
    assert threshold_config.resolve_threshold("mobilenet_v3_small", threshold=.5) == .5


def test_mobilenet_evaluation_uses_selected_threshold(experiment, tmp_path, monkeypatch):
    root, manifest, _, config, checkpoint = experiment
    monkeypatch.setattr(evaluation, "save_roc_pr_plots", lambda *args: None)
    result = evaluation.evaluate_checkpoint(checkpoint, root, manifest, tmp_path / "evaluation",
                                            device="cpu", threshold_config=config, batch_size=2)
    assert result["metrics"]["threshold"] == .4
    assert result["model_type"] == "mobilenet_v3_small"
    assert result["threshold_source"] == "validation_configuration"
    assert result["n_samples"] == 4


def test_mobilenet_api_environment_and_inference_agree(experiment, monkeypatch):
    _, _, _, config, checkpoint = experiment
    monkeypatch.setenv("MOBILENET_V3_SMALL_CHECKPOINT", str(checkpoint))
    monkeypatch.delenv("CUSTOM_CNN_CHECKPOINT", raising=False)
    monkeypatch.delenv("RESNET18_CHECKPOINT", raising=False)
    monkeypatch.delenv("THRESHOLD_CONFIG", raising=False)
    monkeypatch.setattr(threshold_config, "MOBILENET_THRESHOLD_CONFIG", config)
    image = Image.new("L", (40, 30), color=80)
    payload = BytesIO()
    image.save(payload, format="PNG")
    expected = CheckpointPredictor(checkpoint, device="cpu").predict(image)
    with TestClient(api.create_app(device="cpu")) as client:
        assert set(client.get("/health").json()["models"]) == {"mobilenet_v3_small"}
        response = client.post("/predict?model_type=mobilenet_v3_small", files={"file": ("synthetic.png", payload.getvalue())})
        assert response.status_code == 200
        result = response.json()
        assert result["probability_class_1"] == expected["probability_class_1"]
        assert result["prediction"] == expected["prediction"] == 1
        assert result["threshold"] == .4
        assert result["class_name"] == "inhomogeneous"
        assert client.post("/predict?model_type=resnet18", files={"file": ("x", payload.getvalue())}).status_code == 503


def test_mobilenet_frozen_backbone_gradcam_preserves_outputs(experiment):
    _, _, _, _, checkpoint = experiment
    model, _ = load_checkpoint(checkpoint)
    layer_name, layer = get_target_layer(model, "mobilenet_v3_small")
    inputs = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        before = model(inputs)
    heatmap = compute_gradcam(model, inputs, layer, target_class=1)
    assert layer_name == "features.12"
    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()
    with torch.no_grad():
        assert torch.equal(before, model(inputs))
    assert not model.training and not layer._forward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


def test_mobilenet_threshold_selection_cli(experiment, tmp_path, monkeypatch, capsys):
    _, manifest, predictions, _, _ = experiment
    output = tmp_path / "cli_thresholds.json"
    monkeypatch.setattr(sys, "argv", ["selection", "--mobilenet-predictions", str(predictions),
                                     "--validation-manifest", str(manifest), "--output", str(output)])
    threshold_selection.main()
    result = json.loads(capsys.readouterr().out)
    assert result == threshold_config.load_threshold_configuration(output)
    with pytest.raises(FileExistsError):
        threshold_selection.main()


@pytest.fixture
def frozen_experiment(experiment, tmp_path):
    root, manifest, predictions, config, checkpoint = experiment
    hashes = {"validation": exploratory_evaluation.sha256(manifest), "train": "synthetic-training-hash"}
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"best_epoch": 1, "best_validation_loss": .7, "manifest_sha256": hashes}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"original_test_outcomes_previously_observed": True,
                                    "original_artifact_sha256": {}, "manifest_sha256": hashes}))
    frozen = tmp_path / "freeze.json"
    exploratory_evaluation.freeze_experiment(
        checkpoint=checkpoint, threshold_config=config, validation_predictions=predictions,
        validation_manifest=manifest, training_summary=summary, protocol_path=protocol, output=frozen)
    return root, manifest, checkpoint, frozen


def test_frozen_experiment_allows_only_one_evaluation_attempt(frozen_experiment, tmp_path, monkeypatch):
    root, manifest, _, frozen = frozen_experiment
    monkeypatch.setattr(evaluation, "save_roc_pr_plots", lambda *args: None)
    output = tmp_path / "test_evaluation"
    result = exploratory_evaluation.evaluate_once(frozen, root, manifest, output)
    assert result["metrics"]["threshold"] == .4
    receipt = json.loads((output / "evaluation_receipt.json").read_text())
    assert receipt["complete"] and receipt["evaluation_attempt"] == 1
    assert receipt["freeze_sha256"] == exploratory_evaluation.sha256(frozen)
    with pytest.raises(FileExistsError):
        exploratory_evaluation.evaluate_once(frozen, root, manifest, output)


def test_changed_frozen_checkpoint_refuses_evaluation_before_loading_images(frozen_experiment, tmp_path):
    _, manifest, checkpoint, frozen = frozen_experiment
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        exploratory_evaluation.evaluate_once(frozen, tmp_path / "missing-images", manifest, tmp_path / "test")
    assert not (tmp_path / "test").exists()


def test_failed_evaluation_is_reserved_and_cannot_be_retried(frozen_experiment, tmp_path, monkeypatch):
    root, manifest, _, frozen = frozen_experiment
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(exploratory_evaluation, "evaluate_checkpoint", fail)
    output = tmp_path / "attempt"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        exploratory_evaluation.evaluate_once(frozen, root, manifest, output)
    assert not json.loads((output / "evaluation_receipt.json").read_text())["complete"]
    with pytest.raises(FileExistsError):
        exploratory_evaluation.evaluate_once(frozen, root, manifest, output)
