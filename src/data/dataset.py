from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TesticulUSDataset(Dataset):
    """
    PyTorch dataset for TesticulUS binary classification.

    Labels:
        0 = homogeneous
        1 = inhomogeneous

    The dataset does not create train/validation/test splits.
    """

    CLASS_TO_LABEL = {
        "homogeneous": 0,
        "inhomogeneous": 1,
    }

    def __init__(
        self,
        data_dir,
        manifest_path=None,
        transform=None,
    ):
        self.data_dir = Path(data_dir)

        self.transform = transform or transforms.ToTensor()

        if manifest_path is not None:
            self.samples = self._load_manifest(manifest_path)
        else:
            self.samples = self._discover_images()

        if not self.samples:
            raise ValueError("No classification images found.")

    def _discover_images(self):
        samples = []

        for class_name, label in self.CLASS_TO_LABEL.items():

            class_dir = self.data_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing class directory: {class_dir}"
                )

            for image_path in sorted(class_dir.glob("*.png")):

                samples.append({
                    "image_path": image_path,
                    "label": label,
                    "image_id": image_path.stem,
                })

        return samples

    def _load_manifest(self, manifest_path):
        df = pd.read_csv(manifest_path, dtype={"image_id": str})

        required_columns = {"image_id", "label"}

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Manifest missing columns: {missing}"
            )

        samples = []

        for _, row in df.iterrows():

            label = int(row["label"])

            if label not in (0, 1):
                raise ValueError(
                    f"Invalid label: {label}"
                )

            class_name = (
                "homogeneous"
                if label == 0
                else "inhomogeneous"
            )

            image_id = str(row["image_id"])

            image_path = (
                self.data_dir
                / class_name
                / f"{image_id}.png"
            )

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            samples.append({
                "image_path": image_path,
                "label": label,
                "image_id": image_id,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with Image.open(sample["image_path"]) as img:
            image = img.convert("L")

            image = self.transform(image)

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return {
            "image": image,
            "label": label,
            "image_id": sample["image_id"],
        }