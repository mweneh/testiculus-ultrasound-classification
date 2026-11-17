from io import BytesIO
import math

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import torch

from src import api
from src.inference import CheckpointPredictor
from src.models.architectures import CustomCNN, build_resnet18
from src.threshold_config import load_selected_threshold
from src.training import CLASS_TO_LABEL


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    folder = tmp_path_factory.mktemp("untrained-api-checkpoints")
    paths = {}
    for name in ("custom_cnn", "resnet18"):
        torch.manual_seed(42)
        model = CustomCNN() if name == "custom_cnn" else build_resnet18(pretrained=False)
        # A constant probability of 0.4 exercises the actual model-specific
        # threshold decisions: Custom CNN -> 0; ResNet18 -> 1.
        head = model.classifier[-1] if name == "custom_cnn" else model.fc
        with torch.no_grad():
            head.weight.zero_()
            head.bias.copy_(torch.tensor([0.0, math.log(0.4 / 0.6)]))
        path = folder / f"{name}.pth"
        torch.save({
            "model_type": name, "architecture": "CustomCNN" if name == "custom_cnn" else "ResNet18",
            "class_to_idx": CLASS_TO_LABEL, "model_state_dict": model.state_dict(),
        }, path)
        paths[name] = path
    return paths


def image_bytes(mode="L", format="PNG"):
    stream = BytesIO()
    Image.new(mode, (40, 30), color=80 if mode == "L" else (80, 90, 100)).save(stream, format=format)
    return stream.getvalue()


@pytest.fixture
def client(checkpoints):
    with TestClient(api.create_app(checkpoints, device="cpu")) as client:
        yield client


@pytest.mark.parametrize("model_type,expected", [("custom_cnn", 0), ("resnet18", 1)])
@pytest.mark.parametrize("mode,format", [("L", "PNG"), ("RGB", "JPEG")])
def test_upload_matches_cli_preprocessing_and_frozen_threshold(client, checkpoints, model_type, expected, mode, format):
    data = image_bytes(mode, format)
    response = client.post(f"/predict?model_type={model_type}", files={"file": ("sample", data)})
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == expected
    assert result["class_name"] == ("inhomogeneous" if expected else "homogeneous")
    assert result["threshold"] == load_selected_threshold(model_type)
    assert result["threshold_source"] == "validation_configuration"
    assert "threshold_config" not in result
    predictor = CheckpointPredictor(checkpoints[model_type], device="cpu")
    with Image.open(BytesIO(data)) as image:
        expected_result = predictor.predict(image)
    assert result["probability_class_1"] == expected_result["probability_class_1"]
    assert abs(result["probability_class_1"] - 0.4) < 1e-6


def test_models_load_once_and_health_reports_configured_thresholds(monkeypatch, checkpoints):
    loaded = []
    original = api.CheckpointPredictor
    def load_once(path, **kwargs):
        loaded.append(path)
        return original(path, **kwargs)
    monkeypatch.setattr(api, "CheckpointPredictor", load_once)
    with TestClient(api.create_app(checkpoints, device="cpu")) as client:
        assert client.get("/health").json()["status"] == "ready"
        for _ in range(2):
            assert client.post("/predict?model_type=resnet18", files={"file": ("image.png", image_bytes())}).status_code == 200
        assert len(loaded) == 2
        assert client.get("/health").json()["models"]["resnet18"]["threshold"] == load_selected_threshold("resnet18")


@pytest.mark.parametrize("data,status", [(b"", 400), (b"not an image", 400), (image_bytes(format="GIF"), 415)])
def test_invalid_uploads_are_rejected(client, data, status):
    response = client.post("/predict?model_type=custom_cnn", files={"file": ("input.png", data)})
    assert response.status_code == status


def test_upload_and_pixel_limits(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/predict?model_type=custom_cnn", files={"file": ("x", b"x" * 11)}).status_code == 413
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 10)
    assert client.post("/predict?model_type=custom_cnn", files={"file": ("x", image_bytes())}).status_code == 413


def test_missing_file_or_invalid_model_is_rejected(client):
    assert client.post("/predict?model_type=custom_cnn").status_code == 422
    assert client.post("/predict?model_type=unknown", files={"file": ("x", image_bytes())}).status_code == 422


def test_unconfigured_model_is_unavailable(checkpoints):
    with TestClient(api.create_app({"resnet18": checkpoints["resnet18"]}, device="cpu")) as client:
        assert client.post("/predict?model_type=custom_cnn", files={"file": ("x", image_bytes())}).status_code == 503


def test_environment_configuration_loads_checkpoint(monkeypatch, checkpoints):
    monkeypatch.setenv("CUSTOM_CNN_CHECKPOINT", str(checkpoints["custom_cnn"]))
    monkeypatch.delenv("RESNET18_CHECKPOINT", raising=False)
    monkeypatch.setenv("DEVICE", "cpu")
    with TestClient(api.create_app()) as client:
        assert set(client.get("/health").json()["models"]) == {"custom_cnn"}


@pytest.mark.parametrize("case", ["missing", "wrong_model", "bad_config", "empty"])
def test_startup_fails_without_valid_model_and_configuration(checkpoints, tmp_path, case):
    paths = {"custom_cnn": checkpoints["custom_cnn"]}
    kwargs = {}
    if case == "missing":
        paths = {"custom_cnn": tmp_path / "absent.pth"}
    elif case == "wrong_model":
        paths = {"resnet18": checkpoints["custom_cnn"]}
    elif case == "bad_config":
        kwargs["threshold_config"] = tmp_path / "missing.json"
    else:
        paths = {}
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        with TestClient(api.create_app(paths, device="cpu", **kwargs)):
            pass
