from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_transforms(model_type="custom_cnn", training=False):
    """
    Return preprocessing transforms for TesticulUS classification.

    model_type:
        "custom_cnn" -> 1-channel grayscale
        "resnet18"   -> 3-channel RGB-compatible input

    training:
        True  -> apply training augmentation
        False -> deterministic preprocessing
    """

    if model_type not in ("custom_cnn", "resnet18"):
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = [
        transforms.Resize(
            (224, 224),
            interpolation=InterpolationMode.BILINEAR,
        )
    ]

    if training:
        pipeline.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])

    if model_type == "custom_cnn":
        pipeline.extend([
            transforms.ToTensor(),
        ])

    elif model_type == "resnet18":
        pipeline.extend([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose(pipeline)