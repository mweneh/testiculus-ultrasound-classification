import json
import math
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from src.gradcam import compute_gradcam, get_target_layer, main, make_overlay
from src.models.architectures import CustomCNN, build_resnet18
from src.training import CLASS_TO_LABEL


class KnownCAMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1)

    def forward(self, inputs):
        score = self.conv(inputs).mean((1, 2, 3))
        return torch.stack((-score, score), dim=1)


def test_gradcam_matches_analytic_attribution_and_preserves_parameters_and_gradients():
    model = KnownCAMModel()
    model.train()
    original = model.conv.weight.detach().clone()
    model.conv.weight.grad = torch.ones_like(model.conv.weight)
    inputs = torch.tensor([[[[0., 1.], [2., 3.]]]])
    with torch.no_grad():
        heatmap = compute_gradcam(model, inputs, model.conv, target_class=1)
    np.testing.assert_allclose(heatmap, [[0, 1 / 3], [2 / 3, 1]])
    assert model.training
    assert torch.equal(model.conv.weight, original)
    assert torch.equal(model.conv.weight.grad, torch.ones_like(model.conv.weight))
    assert not model.conv._forward_hooks
    assert not inputs.requires_grad


def test_negative_attribution_produces_finite_zero_map_with_frozen_parameters():
    model = KnownCAMModel().eval()
    model.requires_grad_(False)
    heatmap = compute_gradcam(model, torch.ones(1, 1, 4, 4), model.conv, target_class=0)
    assert np.isfinite(heatmap).all()
    assert np.count_nonzero(heatmap) == 0
    assert not model.training
    assert not model.conv.weight.requires_grad


@pytest.mark.parametrize("model_type", ["custom_cnn", "resnet18"])
def test_existing_architectures_have_finite_maps_without_changing_predictions(model_type):
    torch.manual_seed(7)
    model = CustomCNN() if model_type == "custom_cnn" else build_resnet18(pretrained=False)
    model.eval()
    inputs = torch.rand(1, 1 if model_type == "custom_cnn" else 3, 64, 64)
    layer_name, layer = get_target_layer(model, model_type)
    assert layer_name == ("features.12" if model_type == "custom_cnn" else "layer4.1")
    with torch.no_grad():
        before = model(inputs)
    for target in (0, 1):
        heatmap = compute_gradcam(model, inputs, layer, target_class=target)
        assert heatmap.shape == (64, 64)
        assert np.isfinite(heatmap).all()
        assert heatmap.min() >= 0 and heatmap.max() <= 1
    with torch.no_grad():
        after = model(inputs)
    assert torch.equal(before, after)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not layer._forward_hooks


def test_hook_is_removed_and_mode_restored_on_failure():
    model = KnownCAMModel().train()
    def fail(inputs):
        model.conv(inputs)
        raise RuntimeError("failure during forward")
    model.forward = fail
    with pytest.raises(RuntimeError, match="during forward"):
        compute_gradcam(model, torch.ones(1, 1, 2, 2), model.conv)
    assert model.training
    assert not model.conv._forward_hooks


def test_overlay_preserves_image_size_and_zero_map_leaves_pixels_unchanged():
    image = Image.new("L", (40, 30), color=80)
    overlay = make_overlay(image, np.zeros((8, 8), dtype=np.float32))
    assert overlay.size == image.size and overlay.mode == "RGB"
    np.testing.assert_array_equal(np.asarray(overlay), np.asarray(image.convert("RGB")))
    highlighted = make_overlay(image, np.ones((8, 8), dtype=np.float32))
    assert np.asarray(highlighted)[0, 0, 0] > 80


@pytest.mark.parametrize("target", [-1, 2, True])
def test_invalid_target_is_rejected(target):
    model = KnownCAMModel()
    with pytest.raises(ValueError, match="target_class"):
        compute_gradcam(model, torch.ones(1, 1, 2, 2), model.conv, target_class=target)


def test_cli_explains_threshold_selected_class_and_writes_aligned_outputs(tmp_path, monkeypatch, capsys):
    model = build_resnet18(pretrained=False)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor([0.0, math.log(0.4 / 0.6)]))
    checkpoint = tmp_path / "untrained.pth"
    torch.save({
        "model_type": "resnet18", "architecture": "ResNet18",
        "class_to_idx": CLASS_TO_LABEL, "model_state_dict": model.state_dict(),
    }, checkpoint)
    image = tmp_path / "synthetic.png"
    Image.new("L", (40, 30), color=80).save(image)
    output = tmp_path / "explanation"
    monkeypatch.setattr(sys, "argv", [
        "gradcam", "--checkpoint", str(checkpoint), "--image", str(image),
        "--output-dir", str(output),
    ])
    main()
    result = json.loads(capsys.readouterr().out)
    # Probability 0.4 is below argmax's 0.5 boundary, but above the frozen
    # ResNet18 threshold. Explain the threshold-selected positive class.
    assert result["prediction"] == result["gradcam"]["target_class"] == 1
    assert result["gradcam"]["constant_map"]
    assert result == json.loads((output / "prediction.json").read_text())
    with Image.open(output / "heatmap.png") as heatmap:
        assert heatmap.size == (224, 224)
    with Image.open(output / "overlay.png") as overlay:
        assert overlay.size == (40, 30)
