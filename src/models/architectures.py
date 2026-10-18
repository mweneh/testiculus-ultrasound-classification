import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class CustomCNN(nn.Module):
    """
    Lightweight CNN baseline for grayscale
    testicular ultrasound classification.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18(num_classes=2, pretrained=True):
    """
    ResNet18 with a binary classification head.
    """

    weights = (
        ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model