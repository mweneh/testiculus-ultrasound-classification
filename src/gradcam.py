"""Gradient-weighted class activation maps for supported research classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


def get_target_layer(model: torch.nn.Module, model_type: str) -> tuple[str, torch.nn.Module]:
    if model_type == "custom_cnn":
        return "features.12", model.features[12]
    if model_type == "resnet18":
        return "layer4.1", model.layer4[-1]
    if model_type == "mobilenet_v3_small":
        return "features.12", model.features[-1]
    raise ValueError(f"Unsupported Grad-CAM model_type: {model_type}")


def compute_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor,
                    target_layer: torch.nn.Module, target_class: int = 1) -> np.ndarray:
    """Differentiate the target logit, pool gradients, weight features, then ReLU.

    Return a finite, min-max normalized map at the model input resolution.
    Constant/zero maps become zeros. Model weights, parameter gradients and
    original train/eval flags are preserved; the temporary hook is always removed.
    """
    if type(target_class) is not int or target_class not in (0, 1):
        raise ValueError("target_class must be 0 or 1")
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM requires a single BCHW image tensor")
    if target_layer not in model.modules():
        raise ValueError("Target layer must belong to the model")
    modes = {module: module.training for module in model.modules()}
    activations = []
    hook = target_layer.register_forward_hook(lambda module, inputs, output: activations.append(output))
    try:
        model.eval()
        with torch.enable_grad():
            inputs = image_tensor.detach().clone().requires_grad_(True)
            logits = model(inputs)
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite binary classification logits")
            if len(activations) != 1 or activations[0].ndim != 4:
                raise ValueError("Target layer must produce one spatial feature tensor")
            features = activations[0]
            gradients, = torch.autograd.grad(logits[0, target_class], features)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach()
            if not torch.isfinite(cam).all():
                raise ValueError("Grad-CAM produced non-finite values")
            cam = cam - cam.min()
            maximum = cam.max()
            cam = cam / maximum if maximum > 0 else torch.zeros_like(cam)
            return cam.cpu().numpy().astype(np.float32)
    finally:
        hook.remove()
        for module, training in modes.items():
            module.training = training


def make_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay red attribution, resized back to the original image geometry."""
    heatmap = np.asarray(heatmap)
    if (heatmap.ndim != 2 or not np.isfinite(heatmap).all()
            or (heatmap < 0).any() or (heatmap > 1).any() or not 0 <= alpha <= 1):
        raise ValueError("Expected a finite 2D heatmap in [0, 1] and alpha in [0, 1]")
    mask = Image.fromarray(np.rint(heatmap * 255).astype(np.uint8))
    mask = np.asarray(mask.resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255
    opacity = alpha * mask[..., None]
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.array([255, 40, 0], dtype=np.float32)
    return Image.fromarray(np.rint(original * (1 - opacity) + color * opacity).astype(np.uint8))


def main() -> None:
    from src.inference import CheckpointPredictor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold-config", type=Path)
    parser.add_argument("--target-class", type=int, choices=(0, 1))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/gradcam"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    predictor = CheckpointPredictor(args.checkpoint, threshold_config=args.threshold_config, device=args.device)
    with Image.open(args.image) as image:
        result = predictor.predict(image)
        target_class = result["prediction"] if args.target_class is None else args.target_class
        layer_name, layer = get_target_layer(predictor.model, predictor.model_type)
        inputs = predictor.transform(image.convert("L")).unsqueeze(0).to(predictor.device)
        with predictor._lock:
            heatmap = compute_gradcam(predictor.model, inputs, layer, target_class)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(heatmap * 255).astype(np.uint8)).save(args.output_dir / "heatmap.png")
        make_overlay(image, heatmap).save(args.output_dir / "overlay.png")
    result["gradcam"] = {
        "target_class": target_class, "target_layer": layer_name,
        "score": "pre-softmax class logit", "constant_map": bool(np.ptp(heatmap) == 0),
        "input_heatmap_shape": list(heatmap.shape),
    }
    (args.output_dir / "prediction.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
