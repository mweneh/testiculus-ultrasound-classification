import base64
from io import BytesIO
import math

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest
import torch

from src import api
from src.gradcam import compute_gradcam, get_target_layer
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


@pytest.fixture(scope="module")
def random_cnn_checkpoint(tmp_path_factory):
    """Untrained weights giving a nonconstant attribution map (seed probed once)."""
    torch.manual_seed(0)
    model = CustomCNN()
    path = tmp_path_factory.mktemp("random-api-checkpoint") / "custom_cnn.pth"
    torch.save({
        "model_type": "custom_cnn", "architecture": "CustomCNN",
        "class_to_idx": CLASS_TO_LABEL, "model_state_dict": model.state_dict(),
    }, path)
    return path


def textured_png():
    pixels = (np.arange(48 * 64, dtype=np.uint8).reshape(48, 64) * 7) % 251
    stream = BytesIO()
    Image.fromarray(pixels, mode="L").save(stream, format="PNG")
    return stream.getvalue()


def decode_heatmap(result):
    data = base64.b64decode(result["gradcam"]["heatmap_png_base64"], validate=True)
    with Image.open(BytesIO(data)) as image:
        assert image.format == "PNG" and image.mode == "L"
        return np.asarray(image, dtype=np.float32) / 255


@pytest.mark.parametrize("model_type", ["custom_cnn", "resnet18"])
def test_explain_matches_prediction_and_returns_input_resolution_map(client, model_type):
    data = textured_png()
    predicted = client.post(f"/predict?model_type={model_type}", files={"file": ("x.png", data)})
    explained = client.post(f"/explain?model_type={model_type}", files={"file": ("x.png", data)})
    assert predicted.status_code == explained.status_code == 200
    result = explained.json()
    for key, value in predicted.json().items():
        assert result[key] == value
    details = result["gradcam"]
    assert details["target_class"] == result["prediction"]
    assert details["target_layer"] == ("features.12" if model_type == "custom_cnn" else "layer4.1")
    assert details["score"] == "pre-softmax class logit"
    assert details["heatmap_shape"] == [224, 224]
    heatmap = decode_heatmap(result)
    assert heatmap.shape == (224, 224) and np.isfinite(heatmap).all()
    # Zeroed classifier weights make logits feature-independent: the API must
    # report that constant map honestly rather than invent structure.
    assert details["constant_map"] and not heatmap.any()


def test_explain_heatmap_equals_direct_gradcam_and_leaves_predictions_unchanged(random_cnn_checkpoint):
    data = textured_png()
    with TestClient(api.create_app({"custom_cnn": random_cnn_checkpoint}, device="cpu")) as client:
        before = client.post("/predict?model_type=custom_cnn", files={"file": ("x.png", data)}).json()
        result = client.post("/explain?model_type=custom_cnn", files={"file": ("x.png", data)}).json()
        after = client.post("/predict?model_type=custom_cnn", files={"file": ("x.png", data)}).json()
    assert before == after
    assert result["probability_class_1"] == before["probability_class_1"]
    predictor = CheckpointPredictor(random_cnn_checkpoint, device="cpu")
    with Image.open(BytesIO(data)) as image:
        inputs = predictor.transform(image.convert("L")).unsqueeze(0)
    _, layer = get_target_layer(predictor.model, "custom_cnn")
    expected = compute_gradcam(predictor.model, inputs, layer, result["prediction"])
    heatmap = decode_heatmap(result)
    assert not result["gradcam"]["constant_map"] and np.ptp(heatmap) > 0
    np.testing.assert_allclose(heatmap, np.rint(expected * 255) / 255, atol=1e-6)


@pytest.mark.parametrize("data,status", [(b"", 400), (b"not an image", 400), (image_bytes(format="GIF"), 415)])
def test_explain_shares_upload_validation(client, data, status):
    response = client.post("/explain?model_type=custom_cnn", files={"file": ("input.png", data)})
    assert response.status_code == status


def test_explain_limits_and_unconfigured_model(client, checkpoints, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/explain?model_type=custom_cnn", files={"file": ("x", b"x" * 11)}).status_code == 413
    monkeypatch.undo()
    assert client.post("/explain?model_type=unknown", files={"file": ("x", image_bytes())}).status_code == 422
    with TestClient(api.create_app({"resnet18": checkpoints["resnet18"]}, device="cpu")) as single:
        response = single.post("/explain?model_type=custom_cnn", files={"file": ("x", image_bytes())})
        assert response.status_code == 503
        assert response.json()["detail"] == "Model custom_cnn is not configured"


@pytest.fixture
def frontend_dist(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>TesticulUS</title><div id=root></div>")
    (tmp_path / "assets" / "app-abc123.js").write_text("console.log('ok')")
    return tmp_path


def test_frontend_is_served_without_shadowing_api_routes(checkpoints, frontend_dist):
    app = api.create_app({"custom_cnn": checkpoints["custom_cnn"]}, device="cpu", frontend_dir=frontend_dist)
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200 and "TesticulUS" in page.text
        assert page.headers["cache-control"] == "no-cache"
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        asset = client.get("/assets/app-abc123.js")
        assert asset.status_code == 200 and "immutable" in asset.headers["cache-control"]
        assert client.get("/health").json()["status"] == "ready"
        assert client.post("/predict?model_type=custom_cnn", files={"file": ("x", image_bytes())}).status_code == 200
        assert {"/health", "/predict", "/explain"} <= set(client.get("/openapi.json").json()["paths"])
        docs = client.get("/docs")
        assert docs.status_code == 200 and "content-security-policy" not in docs.headers
        assert client.get("/missing.js").status_code == 404


def test_api_only_service_when_no_frontend_build(checkpoints, tmp_path, monkeypatch):
    monkeypatch.delenv("FRONTEND_DIST", raising=False)
    monkeypatch.setattr(api, "DEFAULT_FRONTEND_DIST", tmp_path / "absent")
    with TestClient(api.create_app({"custom_cnn": checkpoints["custom_cnn"]}, device="cpu")) as client:
        assert client.get("/").status_code == 404
        assert client.get("/health").status_code == 200


def test_explicit_frontend_directory_must_contain_index(tmp_path, monkeypatch, frontend_dist):
    with pytest.raises(RuntimeError, match="index.html"):
        api.create_app({}, frontend_dir=tmp_path / "empty")
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="index.html"):
        api.create_app({})
    monkeypatch.setenv("FRONTEND_DIST", str(frontend_dist))
    assert api.resolve_frontend_dir() == frontend_dist
