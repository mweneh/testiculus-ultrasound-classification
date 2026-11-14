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

    __test__ = False

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
        df = pd.read_csv(manifest_path, dtype=str)

        required_columns = {"label"}

        missing = required_columns - set(df.columns)

        if "image_path" not in df.columns and "image_id" not in df.columns:
            missing.add("image_path (or legacy image_id)")

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

            if "image_path" in df.columns and pd.notna(row.get("image_path")):
                relative_path = Path(str(row["image_path"]))
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(
                        f"Manifest image_path must be relative to data_dir: "
                        f"{relative_path}"
                    )
                if relative_path.parts[0] != class_name:
                    raise ValueError(
                        f"Manifest label {label} does not match image_path "
                        f"{relative_path}"
                    )
                image_path = self.data_dir / relative_path
                image_id = image_path.stem
            else:
                # Keep compatibility with earlier manifests containing only
                # image_id and label columns.
                image_id = str(row["image_id"])
                image_path = self.data_dir / class_name / f"{image_id}.png"

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            samples.append({
                "image_path": image_path,
                "relative_path": image_path.relative_to(self.data_dir).as_posix(),
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
            "image_path": sample.get(
                "relative_path",
                sample["image_path"].relative_to(self.data_dir).as_posix(),
            ),
        }
