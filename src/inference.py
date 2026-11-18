"""Single-image inference using a frozen validation-selected threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import RLock

import torch
from PIL import Image

from src.data.transforms import get_transforms
from src.threshold_config import default_threshold_config, resolve_threshold
from src.training import CLASS_TO_LABEL, load_checkpoint, select_device


class CheckpointPredictor:
    """Load a checkpoint and its decision threshold once for repeated inference."""

    def __init__(self, checkpoint_path: Path, *,
                 threshold_config: Path | None = None,
                 device: str | torch.device = "auto") -> None:
        self.device = select_device(device)
        self.model, checkpoint = load_checkpoint(checkpoint_path, device=self.device)
        self.model_type = checkpoint["model_type"]
        if checkpoint.get("class_to_idx") != CLASS_TO_LABEL:
            raise ValueError("Checkpoint class mapping must assign inhomogeneous to class 1")
        self.threshold_config = (
            default_threshold_config(self.model_type) if threshold_config is None else Path(threshold_config)
        )
        self.threshold = resolve_threshold(self.model_type, threshold_config=self.threshold_config)
        self.transform = get_transforms(self.model_type, training=False)
        self._lock = RLock()

    def predict(self, image: Image.Image) -> dict:
        """Predict an image without discovering or opening any dataset partition."""
        tensor = self.transform(image.convert("L")).unsqueeze(0).to(self.device)
        with self._lock, torch.inference_mode():
            probability = float(torch.softmax(self.model(tensor), dim=1)[0, 1].item())
        if not 0 <= probability <= 1:
            raise ValueError("Model returned a non-finite positive-class probability")
        prediction = int(probability >= self.threshold)
        return {
            "model_type": self.model_type,
            "positive_class": 1,
            "probability_class_1": probability,
            "prediction": prediction,
            "class_name": "inhomogeneous" if prediction == 1 else "homogeneous",
            "threshold": self.threshold,
            "threshold_source": "validation_configuration",
            "threshold_config": str(self.threshold_config),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    predictor = CheckpointPredictor(
        args.checkpoint, threshold_config=args.threshold_config, device=args.device
    )
    with Image.open(args.image) as image:
        result = predictor.predict(image)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
