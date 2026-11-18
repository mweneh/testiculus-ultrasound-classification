import torch
import pytest

from src.models.architectures import (
    CustomCNN,
    build_resnet18,
    build_mobilenet_v3_small,
)


def test_custom_cnn():
    model = CustomCNN(num_classes=2)
    model.eval()

    x = torch.randn(2, 1, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()

    print("Custom CNN: PASSED")


def test_resnet18():
    # Avoid downloading pretrained weights
    # during the architecture smoke test.
    model = build_resnet18(
        num_classes=2,
        pretrained=False,
    )
    model.eval()

    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()

    print("ResNet18: PASSED")


def test_mobilenet_v3_small_binary_output_and_factory():
    from src.training import _build_model, _architecture_name, _preprocessing_metadata

    model = _build_model("mobilenet_v3_small", pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.rand(2, 3, 224, 224))
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert model.classifier[-1].out_features == 2
    assert _architecture_name("mobilenet_v3_small") == "MobileNetV3Small"
    metadata = _preprocessing_metadata("mobilenet_v3_small")
    assert metadata["input_channels"] == 3
    assert metadata["evaluation_augmentation"] == []


@pytest.mark.parametrize("pretrained", [False, True])
def test_mobilenet_initialization_option_uses_official_weights_without_test_download(monkeypatch, pretrained):
    from src.models import architectures

    builder = architectures.mobilenet_v3_small
    requested = []
    def build_without_download(*, weights):
        requested.append(weights)
        return builder(weights=None)
    monkeypatch.setattr(architectures, "mobilenet_v3_small", build_without_download)
    model = build_mobilenet_v3_small(pretrained=pretrained)
    expected = architectures.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    assert requested == [expected]
    assert model.classifier[-1].out_features == 2


if __name__ == "__main__":
    test_custom_cnn()
    test_resnet18()

    print("\nAll model architecture tests passed!")
