"""Exercise the Docker service with trained checkpoints and synthetic pixels only."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import time
import uuid

import httpx
from PIL import Image

from src.inference import CheckpointPredictor
from src.threshold_config import LEGACY_MODEL_NAMES as MODEL_NAMES


def _docker(*arguments: str) -> str:
    return subprocess.run(["docker", *arguments], check=True, capture_output=True, text=True).stdout


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_container(image_tag: str, checkpoints_dir: Path, *, include_mobilenet: bool = False) -> dict:
    checkpoints_dir = Path(checkpoints_dir).resolve()
    model_names = (*MODEL_NAMES, "mobilenet_v3_small") if include_mobilenet else MODEL_NAMES
    paths = {name: checkpoints_dir / name / "best_model.pth" for name in model_names}
    image = Image.new("L", (40, 30), color=80)
    stream = BytesIO()
    image.save(stream, format="PNG")
    expected = {}
    hashes = {}
    for name, path in paths.items():
        predictor = CheckpointPredictor(path, device="cpu")
        expected[name] = predictor.predict(image)
        with path.open("rb") as source:
            hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
    container = "testiculus-smoke-" + uuid.uuid4().hex[:8]
    extra_environment = (["-e", "MOBILENET_V3_SMALL_CHECKPOINT=/models/mobilenet_v3_small/best_model.pth"]
                         if include_mobilenet else [])
    _docker(
        "run", "-d", "--rm", "--name", container, "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--health-interval=1s",
        "--health-start-period=5s", "-p", "127.0.0.1::8000",
        "--mount", f"type=bind,src={checkpoints_dir},dst=/models,readonly",
        "-e", "CUSTOM_CNN_CHECKPOINT=/models/custom_cnn/best_model.pth",
        "-e", "RESNET18_CHECKPOINT=/models/resnet18/best_model.pth", *extra_environment, image_tag,
    )
    try:
        details = json.loads(_docker("inspect", container))[0]
        _require(details["Config"]["User"] == "10001:10001", "Unexpected container UID/GID")
        _require(details["HostConfig"]["ReadonlyRootfs"], "Container root filesystem is writable")
        mounts = [m for m in details["Mounts"] if m["Destination"] == "/models"]
        _require(len(mounts) == 1 and not mounts[0]["RW"], "Checkpoint mount is not read-only")
        port = details["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"]
        url = f"http://127.0.0.1:{port}"
        results = {}
        with httpx.Client(timeout=5) as client:
            for _ in range(30):
                try:
                    response = client.get(url + "/health")
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Container did not become ready")
            _require(set(response.json()["models"]) == set(model_names), "Not all trained models are ready")
            for name in model_names:
                response = client.post(url + "/predict", params={"model_type": name}, files={
                    "file": ("synthetic.png", stream.getvalue(), "image/png"),
                })
                response.raise_for_status()
                actual = response.json()
                baseline = expected[name]
                difference = abs(actual["probability_class_1"] - baseline["probability_class_1"])
                _require(difference <= 1e-6, f"{name}: container and local probabilities disagree")
                _require(all(actual[key] == baseline[key] for key in (
                    "prediction", "class_name", "threshold", "threshold_source",
                )), f"{name}: container and local decisions disagree")
                results[name] = {"checkpoint_sha256": hashes[name], "api": actual,
                                 "local_probability_absolute_difference": difference}
            response = client.post(url + "/predict?model_type=custom_cnn", files={"file": ("bad.png", b"bad")})
            _require(response.status_code == 400, "Container accepted an invalid image")
        code = (
            "from pathlib import Path; import os; "
            "assert os.getuid() == 10001; "
            "assert not any(Path('/app').rglob('*.pth')); "
            "assert all(not Path('/app', p).exists() for p in "
            "('data', 'training_runs', 'results', '.git', '.venv', '.env'))"
        )
        _docker("exec", container, "python", "-c", code)
        _docker("exec", container, "python", "-c",
                "from PIL import Image; Image.new('L', (40,30), color=80).save('/tmp/synthetic.png')")
        for name in model_names:
            output = json.loads(_docker(
                "exec", container, "python", "-m", "src.gradcam", "--checkpoint",
                f"/models/{name}/best_model.pth", "--image", "/tmp/synthetic.png",
                "--output-dir", f"/tmp/gradcam/{name}",
            ))
            _require(output["gradcam"]["target_class"] == expected[name]["prediction"],
                     f"{name}: Grad-CAM explained the wrong class")
            _require(output["gradcam"]["input_heatmap_shape"] == [224, 224], "Invalid Grad-CAM geometry")
            results[name]["gradcam"] = output["gradcam"]
        details = json.loads(_docker("inspect", container))[0]
        _require(details["State"]["Health"]["Status"] == "healthy", "Docker health check failed")
        return {
            "format_version": 1, "verified": True, "image_tag": image_tag, "image_id": details["Image"],
            "inputs": "synthetic grayscale image; original trained checkpoints",
            "training_performed": False, "test_images_accessed": False,
            "non_root_user": "10001:10001", "root_filesystem_read_only": True,
            "checkpoints_mounted_read_only": True, "health_check_passed": True,
            "excluded_artifact_directories_absent_from_image": True,
            "invalid_upload_rejected": True, "models": results,
        }
    finally:
        _docker("stop", container)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-tag", default="testiculus-inference:local")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("training_runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/container_verification.json"))
    parser.add_argument("--include-mobilenet", action="store_true")
    args = parser.parse_args()
    report = verify_container(args.image_tag, args.checkpoints_dir, include_mobilenet=args.include_mobilenet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verified": True, "report": str(args.output), "temporary_container_removed": True}))


if __name__ == "__main__":
    main()
