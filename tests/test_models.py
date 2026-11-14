import torch

from src.models.architectures import (
    CustomCNN,
    build_resnet18,
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


if __name__ == "__main__":
    test_custom_cnn()
    test_resnet18()

    print("\nAll model architecture tests passed!")