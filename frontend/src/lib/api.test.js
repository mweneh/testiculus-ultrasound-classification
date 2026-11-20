import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, explain, fetchHealth, parsePrediction, predict } from './api.js';

const json = (status, body) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const prediction = {
  model_type: 'resnet18',
  positive_class: 1,
  probability_class_1: 0.68,
  prediction: 1,
  class_name: 'inhomogeneous',
  threshold: 0.3251684010028839,
  threshold_source: 'validation_configuration',
};

afterEach(() => vi.unstubAllGlobals());

describe('fetchHealth', () => {
  it('reports exactly the models the server loaded', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json(200, { status: 'ready', models: { resnet18: { threshold: 0.3 }, rogue: { threshold: 0.2 } } })));
    await expect(fetchHealth()).resolves.toEqual({ status: 'ready', models: { resnet18: { threshold: 0.3 } } });
  });

  it('surfaces an unreachable service as a network error', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))));
    await expect(fetchHealth()).rejects.toMatchObject({ name: 'ApiError', kind: 'network' });
  });
});

describe('predict', () => {
  it('posts multipart data to the existing endpoint with an explicit model', async () => {
    const fetchMock = vi.fn(async () => json(200, prediction));
    vi.stubGlobal('fetch', fetchMock);
    const upload = new File(['png'], 'scan.png', { type: 'image/png' });
    const result = await predict('resnet18', upload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/predict?model_type=resnet18');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.body.get('file').name).toBe('scan.png');
    expect(result).toEqual({
      modelType: 'resnet18',
      score: 0.68,
      prediction: 1,
      className: 'inhomogeneous',
      threshold: 0.3251684010028839,
      thresholdSource: 'validation_configuration',
    });
  });

  it.each([
    [408, 'Upload timed out', /upload timed out/],
    [429, 'Research demo rate limit reached', /Wait one minute/],
    [413, 'Image exceeds the 5 MiB upload limit', /server limits/],
    [415, 'Only PNG and JPEG images are supported', /Only PNG and JPEG/],
    [503, 'Model mobilenet_v3_small is not configured', /not loaded on the server/],
    [500, null, /HTTP 500/],
  ])('maps HTTP %i to a readable error and keeps the server detail', async (status, detail, message) => {
    vi.stubGlobal('fetch', vi.fn(async () => json(status, detail ? { detail } : {})));
    const error = await predict('resnet18', new File(['x'], 'x.png')).catch((caught) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(status);
    expect(error.message).toMatch(message);
    expect(error.detail).toBe(detail);
  });

  it('never accepts a response that does not match the prediction contract', () => {
    expect(() => parsePrediction({ ...prediction, model_type: 'custom_cnn' }, 'resnet18')).toThrow(ApiError);
    expect(() => parsePrediction({ ...prediction, probability_class_1: 1.2 }, 'resnet18')).toThrow(ApiError);
    expect(() => parsePrediction({ ...prediction, class_name: 'homogeneous' }, 'resnet18')).toThrow(ApiError);
    expect(() => parsePrediction({ ...prediction, threshold_source: 'default' }, 'resnet18')).toThrow(ApiError);
  });
});

describe('explain', () => {
  it('returns the prediction with Grad-CAM details from /explain', async () => {
    const fetchMock = vi.fn(async () =>
      json(200, {
        ...prediction,
        gradcam: {
          target_class: 1,
          target_layer: 'layer4.1',
          score: 'pre-softmax class logit',
          constant_map: false,
          heatmap_shape: [224, 224],
          heatmap_png_base64: 'iVBORw0KGgo=',
        },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const result = await explain('resnet18', new File(['x'], 'x.png'));
    expect(fetchMock.mock.calls[0][0]).toBe('/explain?model_type=resnet18');
    expect(result.gradcam).toMatchObject({ targetClass: 1, targetLayer: 'layer4.1', constantMap: false, shape: [224, 224] });
  });

  it('rejects a response without a heatmap', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json(200, prediction)));
    await expect(explain('resnet18', new File(['x'], 'x.png'))).rejects.toThrow(/valid Grad-CAM map/);
  });
});
