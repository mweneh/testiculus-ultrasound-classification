import torch
import json
import pytest
import sys
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from src.models.architectures import CustomCNN, build_mobilenet_v3_small
from src.training import (
    compute_class_weights,
    load_checkpoint,
    train_model,
)


class SyntheticClassificationDataset(Dataset):
    def __init__(self, labels, channels=1, size=16):
        self.labels = labels
        self.images = torch.rand(len(labels), channels, size, size, generator=torch.Generator().manual_seed(7))

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


def test_mobilenet_frozen_backbone_training_checkpoint_and_timing(tmp_path):
    model = build_mobilenet_v3_small(pretrained=False)
    backbone = {key: tensor.clone() for key, tensor in model.features.state_dict().items()}
    head = model.classifier[-1].weight.detach().clone()
    dataset = SyntheticClassificationDataset([0, 1, 0, 1], channels=3, size=64)
    loader = DataLoader(dataset, batch_size=2)
    history, checkpoint = train_model(
        model, loader, loader, tmp_path, model_type="mobilenet_v3_small",
        class_weights=compute_class_weights(dataset.labels), max_epochs=1, patience=1,
        learning_rate=1e-4, device="cpu", freeze_backbone=True,
    )
    assert all(torch.equal(tensor, model.features.state_dict()[key]) for key, tensor in backbone.items())
    assert not torch.equal(head, model.classifier[-1].weight)
    assert all(not p.requires_grad and p.grad is None for p in model.features.parameters())
    restored, metadata = load_checkpoint(checkpoint)
    assert all(not p.requires_grad for p in restored.features.parameters())
    with torch.no_grad():
        assert torch.equal(model.eval()(dataset.images[:1]), restored(dataset.images[:1]))
    assert metadata["class_to_idx"] == {"homogeneous": 0, "inhomogeneous": 1}
    assert metadata["preprocessing"]["input_channels"] == 3
    summary = json.loads((tmp_path / "training_summary.json").read_text())
    assert summary["device"] == "cpu"
    assert summary["training_seconds"] >= history[0]["epoch_seconds"] > 0
    assert 0 < summary["trainable_parameters"] < summary["parameters"]
    assert metadata["training_config"]["freeze_backbone"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        train_model(model, loader, loader, tmp_path, model_type="mobilenet_v3_small",
                    class_weights=torch.ones(2), max_epochs=1, patience=1)


def test_mobilenet_cli_defaults_to_requested_cpu_experiment(tmp_path, monkeypatch):
    from src import training

    requested = {}
    class Dataset(SyntheticClassificationDataset):
        def __init__(self, *args, **kwargs):
            super().__init__([0, 0, 0, 1], channels=3, size=64)
            self.samples = [{"label": label} for label in self.labels]
    def fake_train(model, train, validation, output, **kwargs):
        requested.update(kwargs)
        requested["batch_size"] = train.batch_size
        Path(output).mkdir()
        (Path(output) / "training_summary.json").write_text("{}")
        return [], Path(output) / "best_model.pth"
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "validation"):
        (splits / f"{name}.csv").write_text("synthetic manifest bytes")
    monkeypatch.setattr(training, "TesticulUSDataset", Dataset)
    monkeypatch.setattr(training, "train_model", fake_train)
    monkeypatch.setattr(training, "_build_model", lambda *args, **kwargs: torch.nn.Linear(1, 2))
    monkeypatch.setattr(sys, "argv", ["training", "--model", "mobilenet_v3_small",
                        "--split-dir", str(splits), "--output-dir", str(tmp_path / "run")])
    training.main()
    assert requested["pretrained"] is True
    assert requested["freeze_backbone"] is True
    assert requested["device"] == "cpu"
    assert requested["max_epochs"] == 20 and requested["patience"] == 5
    assert requested["batch_size"] == 16 and requested["learning_rate"] == 1e-4
    assert torch.allclose(requested["class_weights"], compute_class_weights([0, 0, 0, 1]))
