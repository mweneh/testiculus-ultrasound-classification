"""Comparable CPU model-forward timings and an optional synthetic training probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torchvision

from src.training import _build_model, load_checkpoint, seed_everything


def hardware_information() -> dict:
    cpuinfo = Path("/proc/cpuinfo")
    cpu_model = platform.processor()
    if cpuinfo.exists():
        cpu_model = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    return {
        "platform": platform.platform(), "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "available_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
        "python_version": platform.python_version(), "device": "cpu",
        "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
    }


def benchmark_forward(model, inputs, *, warmup=10, repeats=50) -> dict:
    """Time batch-one CPU forwards; exclude image decoding and preprocessing."""
    if warmup < 1 or repeats < 2 or inputs.device.type != "cpu":
        raise ValueError("Require CPU inputs, positive warmup and at least two repeats")
    modes = {module: module.training for module in model.modules()}
    measurements = []
    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                model(inputs)
            for _ in range(repeats):
                started = time.perf_counter()
                model(inputs)
                measurements.append((time.perf_counter() - started) * 1000)
    finally:
        for module, training in modes.items():
            module.training = training
    return {
        "median_ms": float(np.median(measurements)),
        "mean_ms": float(np.mean(measurements)),
        "p95_ms": float(np.percentile(measurements, 95)),
        "min_ms": min(measurements), "measurements_ms": measurements,
    }


def probe_head_training(model, *, batch_size=16, warmup=1, repeats=3) -> dict:
    """Optimize only a disposable copy on random tensors, never a checkpoint."""
    if batch_size < 1 or warmup < 1 or repeats < 1:
        raise ValueError("Training probe sizes must be positive")
    probe = copy.deepcopy(model).cpu().train()
    probe.features.requires_grad_(False)
    probe.features.eval()
    optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    inputs = torch.rand(batch_size, 3, 224, 224)
    labels = torch.arange(batch_size) % 2
    measurements = []
    for index in range(warmup + repeats):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        criterion(probe(inputs), labels).backward()
        optimizer.step()
        if index >= warmup:
            measurements.append(time.perf_counter() - started)
    return {"batch_size": batch_size, "warmup_steps": warmup, "measured_steps": repeats,
            "median_step_seconds": float(np.median(measurements)), "step_seconds": measurements,
            "inputs": "synthetic tensors; excludes image loading/augmentation",
            "disposable_model_copy": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-cnn-checkpoint", type=Path, default=Path("training_runs/custom_cnn/best_model.pth"))
    parser.add_argument("--resnet18-checkpoint", type=Path, default=Path("training_runs/resnet18/best_model.pth"))
    parser.add_argument("--mobilenet-checkpoint", type=Path)
    parser.add_argument("--pretrained", action="store_true", help="Initialize MobileNet from ImageNet if no checkpoint supplied")
    parser.add_argument("--probe-training", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 1 or args.repeats < 2:
        parser.error("Require positive threads/warmup and at least two repeats")
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing benchmark")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_everything(42)
    report = {"hardware": hardware_information(), "conditions": {
        "device": "cpu", "dtype": "float32", "batch_size": 1, "input_size": [224, 224],
        "warmup": args.warmup, "repeats": args.repeats, "seed": 42,
        "scope": "model forward under inference_mode; excludes decoding, preprocessing and API",
        "image_source": "same synthetic grayscale tensor; model-specific channels/normalization",
        "test_images_accessed": False,
    }, "models": {}}
    gray = torch.rand(1, 1, 224, 224)
    rgb = (gray.repeat(1, 3, 1, 1) - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
    paths = {"custom_cnn": args.custom_cnn_checkpoint, "resnet18": args.resnet18_checkpoint,
             "mobilenet_v3_small": args.mobilenet_checkpoint}
    for name, path in paths.items():
        if path is None:
            model = _build_model(name, pretrained=args.pretrained).cpu().eval()
            model.features.requires_grad_(False)
        else:
            model, metadata = load_checkpoint(path, device="cpu")
            if metadata["model_type"] != name:
                raise ValueError(f"Wrong checkpoint for {name}")
        record = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "state_dict_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
            "checkpoint_bytes": path.stat().st_size if path else None,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
            "weight_status": "trained checkpoint" if path else "ImageNet initialization with random binary output" if args.pretrained else "random initialization",
            "input_channels": 1 if name == "custom_cnn" else 3,
            "latency": benchmark_forward(model, gray if name == "custom_cnn" else rgb,
                                         warmup=args.warmup, repeats=args.repeats),
        }
        if name == "mobilenet_v3_small" and args.probe_training:
            record["synthetic_head_training_probe"] = probe_head_training(model)
        report["models"][name] = record
        print(f"{name}: {record['parameters']} parameters; median {record['latency']['median_ms']:.3f} ms", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Benchmark saved to {args.output}")


if __name__ == "__main__":
    main()
