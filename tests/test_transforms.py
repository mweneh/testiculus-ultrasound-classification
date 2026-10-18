import torch

from PIL import Image

from src.data.transforms import get_transforms


def test_custom_cnn_transforms():
    image = Image.new("L", (300, 300), color=128)

    for training in (True, False):
        transform = get_transforms(
            model_type="custom_cnn",
            training=training,
        )

        tensor = transform(image)

        assert tensor.shape == (1, 224, 224)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert tensor.min() >= 0
        assert tensor.max() <= 1


def test_resnet18_transforms():
    image = Image.new("L", (300, 300), color=128)

    for training in (True, False):
        transform = get_transforms(
            model_type="resnet18",
            training=training,
        )

        tensor = transform(image)

        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_evaluation_is_deterministic():
    image = Image.new("L", (300, 300), color=128)

    for model_type in ("custom_cnn", "resnet18"):
        transform = get_transforms(
            model_type=model_type,
            training=False,
        )

        first = transform(image)
        second = transform(image)

        assert torch.equal(first, second)


if __name__ == "__main__":
    test_custom_cnn_transforms()
    test_resnet18_transforms()
    test_evaluation_is_deterministic()

    print("All preprocessing tests passed!")