import torch
from torch.utils.data import DataLoader, Dataset

from src.models.architectures import CustomCNN
from src.training import (
    compute_class_weights,
    load_checkpoint,
    train_model,
)


class SyntheticClassificationDataset(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.images = torch.rand(len(labels), 1, 16, 16, generator=torch.Generator().manual_seed(7))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": torch.tensor(self.labels[index])}


def test_class_weights_use_inverse_training_frequencies():
    weights = compute_class_weights([0, 0, 0, 1])
    assert torch.allclose(weights, torch.tensor([2 / 3, 2.0]))


def test_class_weights_reject_missing_training_class():
    try:
        compute_class_weights([0, 0])
    except ValueError as error:
        assert "missing class" in str(error)
    else:
        raise AssertionError("missing training class was accepted")


def test_synthetic_training_and_checkpoint_roundtrip(tmp_path):
    train_dataset = SyntheticClassificationDataset([0, 1, 0, 1])
    validation_dataset = SyntheticClassificationDataset([0, 1])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=2, shuffle=False)
    model = CustomCNN(num_classes=2)
    sample = validation_dataset[0]["image"].unsqueeze(0)

    history, checkpoint_path = train_model(
        model,
        train_loader,
        validation_loader,
        tmp_path,
        model_type="custom_cnn",
        class_weights=compute_class_weights(train_dataset.labels),
        max_epochs=1,
        patience=1,
        device="cpu",
    )

    assert len(history) == 1
    assert checkpoint_path.is_file()
    assert (tmp_path / "history.csv").is_file()
    model.eval()
    with torch.no_grad():
        trained_output = model(sample)
    restored, metadata = load_checkpoint(checkpoint_path, device="cpu")
    with torch.no_grad():
        restored_output = restored(sample)
    assert torch.allclose(restored_output, trained_output, atol=1e-6)
    assert metadata["architecture"] == "CustomCNN"
    assert metadata["preprocessing"]["input_size"] == [224, 224]
    assert metadata["class_to_idx"] == {"homogeneous": 0, "inhomogeneous": 1}
