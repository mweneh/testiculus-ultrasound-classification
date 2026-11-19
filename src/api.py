"""FastAPI serving for explicit, frozen research checkpoints."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from src.gradcam import compute_gradcam, get_target_layer
from src.inference import CheckpointPredictor
from src.threshold_config import MODEL_NAMES


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ModelType = Literal["custom_cnn", "resnet18", "mobilenet_v3_small"]


class PredictionResponse(BaseModel):
    model_type: ModelType
    positive_class: Literal[1]
    probability_class_1: float = Field(ge=0, le=1, allow_inf_nan=False)
    prediction: Literal[0, 1]
    class_name: Literal["homogeneous", "inhomogeneous"]
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    threshold_source: Literal["validation_configuration"]


class GradCAMDetails(BaseModel):
    target_class: Literal[0, 1]
    target_layer: str
    score: Literal["pre-softmax class logit"]
    constant_map: bool
    heatmap_shape: tuple[int, int]
    heatmap_png_base64: str


class ExplanationResponse(PredictionResponse):
    gradcam: GradCAMDetails


def _read_upload(file: UploadFile) -> Image.Image:
    """Apply the shared upload limits and return a decoded, detached image."""
    payload = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 5 MiB upload limit")
    if not payload:
        raise HTTPException(status_code=400, detail="Image upload is empty")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in ("PNG", "JPEG"):
                raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image exceeds the 20 megapixel limit")
            image.load()
            return image.copy()
    except Image.DecompressionBombError as error:
        raise HTTPException(status_code=413, detail="Image dimensions are too large") from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid or undecodable image") from error


def explain_prediction(predictor: CheckpointPredictor, image: Image.Image) -> dict:
    """Predict, then attribute the frozen-threshold class with Grad-CAM.

    Mirrors the ``src.gradcam`` CLI defaults. Prediction and attribution run
    under the predictor's lock so both describe the same model state. The
    heatmap is the 224x224 model-input map, encoded as an 8-bit grayscale PNG.
    """
    with predictor._lock:
        result = predictor.predict(image)
        target_class = result["prediction"]
        layer_name, layer = get_target_layer(predictor.model, predictor.model_type)
        inputs = predictor.transform(image.convert("L")).unsqueeze(0).to(predictor.device)
        heatmap = compute_gradcam(predictor.model, inputs, layer, target_class)
    stream = BytesIO()
    Image.fromarray(np.rint(heatmap * 255).astype(np.uint8)).save(stream, format="PNG")
    result["gradcam"] = {
        "target_class": target_class,
        "target_layer": layer_name,
        "score": "pre-softmax class logit",
        "constant_map": bool(np.ptp(heatmap) == 0),
        "heatmap_shape": list(heatmap.shape),
        "heatmap_png_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
    }
    return result


def create_app(checkpoint_paths: dict[str, Path] | None = None, *,
               threshold_config: Path | None = None,
               device: str | None = None) -> FastAPI:
    """Load configured models once at startup; never substitute random weights."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        paths = checkpoint_paths
        if paths is None:
            paths = {
                name: Path(value) for name in MODEL_NAMES
                if (value := os.environ.get(f"{name.upper()}_CHECKPOINT"))
            }
        if not paths or not set(paths).issubset(MODEL_NAMES):
            raise RuntimeError(
                "Configure a supported model's CHECKPOINT environment variable with an existing checkpoint"
            )
        config = threshold_config
        if config is None and os.environ.get("THRESHOLD_CONFIG"):
            config = Path(os.environ["THRESHOLD_CONFIG"])
        target_device = device if device is not None else os.environ.get("DEVICE", "cpu")
        predictors = {}
        for name, path in paths.items():
            predictor = CheckpointPredictor(path, threshold_config=config, device=target_device)
            if predictor.model_type != name:
                raise ValueError(f"Checkpoint configured as {name} contains {predictor.model_type}")
            predictors[name] = predictor
        application.state.predictors = predictors
        try:
            yield
        finally:
            application.state.predictors.clear()

    application = FastAPI(
        title="TesticulUS research inference", version="1.0.0",
        description="Frozen image-level research classifiers. Not clinically validated or suitable for diagnosis.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ready",
            "models": {name: {"threshold": predictor.threshold}
                       for name, predictor in application.state.predictors.items()},
        }

    def configured_predictor(model_type: str) -> CheckpointPredictor:
        predictor = application.state.predictors.get(model_type)
        if predictor is None:
            raise HTTPException(status_code=503, detail=f"Model {model_type} is not configured")
        return predictor

    @application.post("/predict", response_model=PredictionResponse)
    def predict(model_type: ModelType, file: UploadFile = File(...)) -> dict:
        predictor = configured_predictor(model_type)
        with _read_upload(file) as decoded:
            return predictor.predict(decoded)

    @application.post("/explain", response_model=ExplanationResponse)
    def explain(model_type: ModelType, file: UploadFile = File(...)) -> dict:
        """Return the frozen-threshold prediction with a coarse Grad-CAM map.

        The map is a model-sensitivity visualization, not a segmentation,
        lesion localization or evidence of clinical correctness.
        """
        predictor = configured_predictor(model_type)
        with _read_upload(file) as decoded:
            return explain_prediction(predictor, decoded)

    return application


app = create_app()
