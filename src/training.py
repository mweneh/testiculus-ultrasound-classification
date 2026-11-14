"""Reproducible training for the custom CNN and ResNet18 classifiers."""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import TesticulUSDataset
from src.data.transforms import get_transforms
from src.models.architectures import CustomCNN, build_resnet18


CLASS_TO_LABEL = {"homogeneous": 0, "inhomogeneous": 1}
NUM_CLASSES = 2


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_class_weights(labels, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Compute inverse-frequency cross-entropy weights from training labels."""
    labels = torch.as_tensor(list(labels), dtype=torch.long)
    if labels.numel() == 0:
        raise ValueError("Cannot compute class weights from no labels")
    if (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError(f"Labels must be in [0, {num_classes - 1}]")
    counts = torch.bincount(labels, minlength=num_classes).to(torch.float32)
    if (counts == 0).any():
        absent = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training data is missing class labels: {absent}")
    return labels.numel() / (num_classes * counts)


def _build_model(model_type: str, pretrained: bool = False) -> nn.Module:
    if model_type == "custom_cnn":
        if pretrained:
            raise ValueError("The custom CNN does not support pretrained weights")
        return CustomCNN(num_classes=NUM_CLASSES)
    if model_type == "resnet18":
        return build_resnet18(num_classes=NUM_CLASSES, pretrained=pretrained)
    raise ValueError(f"Unsupported model type: {model_type}")


def _architecture_name(model_type: str) -> str:
    return {"custom_cnn": "CustomCNN", "resnet18": "ResNet18"}[model_type]


def _preprocessing_metadata(model_type: str) -> dict:
    metadata = {
        "model_type": model_type,
        "input_size": [224, 224],
        "resize": "bilinear",
        "source_image_mode": "grayscale",
        "value_range": [0.0, 1.0],
        "input_channels": 1 if model_type == "custom_cnn" else 3,
        "training_augmentation": [
            "RandomHorizontalFlip(p=0.5)",
            "RandomRotation(degrees=10)",
        ],
        "evaluation_augmentation": [],
    }
    if model_type == "resnet18":
        metadata.update({
            "grayscale_conversion": "3-channel",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        })
    return metadata


def _run_epoch(model: nn.Module, loader: DataLoader,
               criterion: nn.Module, device: torch.device,
               optimizer: torch.optim.Optimizer | None = None) -> dict:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    sample_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            loss_total += loss.detach().item() * batch_size
            correct += (logits.detach().argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("A training or validation loader is empty")
    return {
        "loss": loss_total / sample_count,
        "accuracy": correct / sample_count,
    }


def _write_history(history: list[dict], output_path: Path) -> None:
    fields = (
        "epoch", "train_loss", "train_accuracy", "validation_loss",
        "validation_accuracy", "learning_rate",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def _save_checkpoint(path: Path, model: nn.Module, model_type: str,
                     pretrained: bool, seed: int, epoch: int,
                     validation_loss: float, class_weights: torch.Tensor,
                     training_config: dict) -> None:
    checkpoint = {
        "format_version": 1,
        "model_type": model_type,
        "architecture": _architecture_name(model_type),
        "num_classes": NUM_CLASSES,
        "model_state_dict": {
            key: tensor.detach().cpu()
            for key, tensor in model.state_dict().items()
        },
        "pretrained_initialization": bool(pretrained),
        "class_to_idx": CLASS_TO_LABEL,
        "preprocessing": _preprocessing_metadata(model_type),
        "seed": int(seed),
        "best_epoch": int(epoch),
        "best_validation_loss": float(validation_loss),
        "class_weights": class_weights.detach().cpu().tolist(),
        "training_config": training_config,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output_dir: Path,
    *,
    model_type: str,
    class_weights: torch.Tensor,
    seed: int = 42,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 7,
    min_delta: float = 0.0,
    scheduler_patience: int = 2,
    device: str | torch.device = "auto",
    pretrained: bool = False,
) -> tuple[list[dict], Path]:
    """Train with weighted cross-entropy and validation-loss early stopping."""
    if model_type not in ("custom_cnn", "resnet18"):
        raise ValueError(f"Unsupported model type: {model_type}")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if learning_rate <= 0 or weight_decay < 0 or min_delta < 0:
        raise ValueError("Invalid optimizer or early-stopping configuration")

    seed_everything(seed)
    target_device = select_device(device)
    model.to(target_device)
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
    if (
        class_weights.shape != (NUM_CLASSES,)
        or not torch.isfinite(class_weights).all()
        or (class_weights <= 0).any()
    ):
        raise ValueError("class_weights must contain two positive finite values")
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(target_device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_config = {
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_patience": scheduler_patience,
        "device": str(target_device),
    }

    for epoch in range(1, max_epochs + 1):
        train_stats = _run_epoch(
            model, train_loader, criterion, target_device, optimizer
        )
        validation_stats = _run_epoch(
            model, validation_loader, criterion, target_device
        )
        scheduler.step(validation_stats["loss"])
        learning_rate_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "validation_loss": validation_stats["loss"],
            "validation_accuracy": validation_stats["accuracy"],
            "learning_rate": learning_rate_now,
        })

        if validation_stats["loss"] < best_validation_loss - min_delta:
            best_validation_loss = validation_stats["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path, model, model_type, pretrained, seed, epoch,
                best_validation_loss, class_weights, training_config,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    _write_history(history, output_dir / "history.csv")
    return history, checkpoint_path


def load_checkpoint(checkpoint_path: Path,
                    device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a supported model and load its checkpoint without downloads."""
    target_device = select_device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=target_device, weights_only=True
    )
    model_type = checkpoint.get("model_type")
    if checkpoint.get("architecture") != _architecture_name(model_type):
        raise ValueError("Checkpoint architecture metadata is inconsistent")
    # Checkpoint tensors include the full trained ResNet weights, so loading
    # never needs to download ImageNet initialization again.
    model = _build_model(model_type, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target_device)
    model.eval()
    return model, checkpoint


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("custom_cnn", "resnet18"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    if args.pretrained and args.model != "resnet18":
        parser.error("--pretrained is only available with --model resnet18")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers nonnegative")

    seed_everything(args.seed)
    train_dataset = TesticulUSDataset(
        args.data_dir,
        args.split_dir / "train.csv",
        transform=get_transforms(args.model, training=True),
    )
    validation_dataset = TesticulUSDataset(
        args.data_dir,
        args.split_dir / "validation.csv",
        transform=get_transforms(args.model, training=False),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_seed_worker,
    )
    train_labels = [sample["label"] for sample in train_dataset.samples]
    class_weights = compute_class_weights(train_labels)
    model = _build_model(args.model, pretrained=args.pretrained)
    history, checkpoint_path = train_model(
        model,
        train_loader,
        validation_loader,
        args.output_dir,
        model_type=args.model,
        class_weights=class_weights,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        pretrained=args.pretrained,
    )
    print(f"Training stopped after {len(history)} epochs.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"History: {args.output_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
