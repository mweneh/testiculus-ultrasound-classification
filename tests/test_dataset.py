from pathlib import Path
from collections import Counter

import torch

from torch.utils.data import DataLoader

from src.data.dataset import TesticulUSDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main():

    dataset = TesticulUSDataset(
        data_dir=DATA_DIR
    )

    print("Dataset size:", len(dataset))

    assert len(dataset) == 860, (
        f"Expected 860 images, found {len(dataset)}"
    )

    labels = Counter(
        sample["label"]
        for sample in dataset.samples
    )

    print("Class distribution:", labels)

    assert labels[0] == 696
    assert labels[1] == 164

    sample = dataset[0]

    print("\nSingle sample:")
    print("Image shape:", sample["image"].shape)
    print("Label:", sample["label"].item())
    print("Image ID:", sample["image_id"])

    assert isinstance(sample["image"], torch.Tensor)

    assert sample["image"].shape == (1, 300, 300)

    assert sample["image"].dtype == torch.float32

    assert 0 <= sample["image"].min()
    assert sample["image"].max() <= 1

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\nBatch:")
    print("Images:", batch["image"].shape)
    print("Labels:", batch["label"].shape)
    print("IDs:", batch["image_id"][:3])

    assert batch["image"].shape == (8, 1, 300, 300)
    assert batch["label"].shape == (8,)

    print("\nAll dataset checks passed!")


if __name__ == "__main__":
    main()