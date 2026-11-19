/**
 * Client for the existing FastAPI service (src/api.py).
 *
 *   GET  /health                       -> { status, models: { <model>: { threshold } } }
 *   POST /predict?model_type=<model>   multipart "file" -> PredictionResponse
 *   POST /explain?model_type=<model>   multipart "file" -> PredictionResponse + gradcam
 *
 * Errors use FastAPI's { detail } body. Nothing here fabricates a result: a
 * response is returned only when the server replied 2xx with the expected shape.
 */

export const MODEL_IDS = ['custom_cnn', 'resnet18', 'mobilenet_v3_small'];

export class ApiError extends Error {
  constructor(message, { status = null, kind = 'http', detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.kind = kind;
    this.detail = detail;
  }
}

const STATUS_MESSAGES = {
  400: 'The server could not decode this image. Upload a valid PNG or JPEG file.',
  413: 'The image exceeds the server limits (5 MiB and 20 megapixels).',
  415: 'Only PNG and JPEG images are supported.',
  422: 'The request was incomplete. Select a model and an image, then try again.',
  503: 'This model is not loaded on the server, so it cannot run.',
};

async function readDetail(response) {
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((item) => item?.msg).filter(Boolean).join('; ');
  } catch {
    // Non-JSON error bodies fall back to the status message.
  }
  return null;
}

async function request(path, init) {
  let response;
  try {
    response = await fetch(path, { headers: { Accept: 'application/json' }, ...init });
  } catch (error) {
    if (error?.name === 'AbortError') throw error;
    throw new ApiError('Could not reach the inference API. Check that the service is running.', {
      kind: 'network',
    });
  }
  if (!response.ok) {
    const detail = await readDetail(response);
    const message = STATUS_MESSAGES[response.status] ?? `The server returned an error (HTTP ${response.status}).`;
    throw new ApiError(message, { status: response.status, detail });
  }
  try {
    return await response.json();
  } catch {
    throw new ApiError('The server returned an unreadable response.', { status: response.status, kind: 'parse' });
  }
}

const isUnit = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;

export function parsePrediction(body, expectedModel) {
  const valid =
    body &&
    body.model_type === expectedModel &&
    isUnit(body.probability_class_1) &&
    isUnit(body.threshold) &&
    (body.prediction === 0 || body.prediction === 1) &&
    body.class_name === (body.prediction === 1 ? 'inhomogeneous' : 'homogeneous') &&
    body.threshold_source === 'validation_configuration';
  if (!valid) {
    throw new ApiError('The server response did not match the expected prediction format.', { kind: 'parse' });
  }
  return {
    modelType: body.model_type,
    score: body.probability_class_1,
    prediction: body.prediction,
    className: body.class_name,
    threshold: body.threshold,
    thresholdSource: body.threshold_source,
  };
}

export async function fetchHealth({ signal } = {}) {
  const body = await request('/health', { signal });
  if (body?.status !== 'ready' || typeof body.models !== 'object' || body.models === null) {
    throw new ApiError('The health endpoint returned an unexpected response.', { kind: 'parse' });
  }
  const models = {};
  for (const id of MODEL_IDS) {
    const threshold = body.models[id]?.threshold;
    if (isUnit(threshold)) models[id] = { threshold };
  }
  return { status: body.status, models };
}

function uploadForm(file) {
  const form = new FormData();
  form.append('file', file, file.name || 'upload');
  return form;
}

export async function predict(modelType, file, { signal } = {}) {
  const body = await request(`/predict?model_type=${encodeURIComponent(modelType)}`, {
    method: 'POST',
    body: uploadForm(file),
    signal,
  });
  return parsePrediction(body, modelType);
}

export async function explain(modelType, file, { signal } = {}) {
  const body = await request(`/explain?model_type=${encodeURIComponent(modelType)}`, {
    method: 'POST',
    body: uploadForm(file),
    signal,
  });
  const prediction = parsePrediction(body, modelType);
  const gradcam = body.gradcam;
  const shape = gradcam?.heatmap_shape;
  if (
    !gradcam ||
    (gradcam.target_class !== 0 && gradcam.target_class !== 1) ||
    typeof gradcam.target_layer !== 'string' ||
    typeof gradcam.heatmap_png_base64 !== 'string' ||
    !Array.isArray(shape) ||
    shape.length !== 2
  ) {
    throw new ApiError('The server response did not include a valid Grad-CAM map.', { kind: 'parse' });
  }
  return {
    ...prediction,
    gradcam: {
      targetClass: gradcam.target_class,
      targetLayer: gradcam.target_layer,
      score: gradcam.score,
      constantMap: Boolean(gradcam.constant_map),
      shape,
      pngBase64: gradcam.heatmap_png_base64,
    },
  };
}
