import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App.jsx';

const THRESHOLDS = { custom_cnn: 0.4541136622428894, resnet18: 0.3251684010028839, mobilenet_v3_small: 0.5038942694664001 };

const json = (status, body) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

const predictionBody = (model, score) => ({
  model_type: model,
  positive_class: 1,
  probability_class_1: score,
  prediction: score >= THRESHOLDS[model] ? 1 : 0,
  class_name: score >= THRESHOLDS[model] ? 'inhomogeneous' : 'homogeneous',
  threshold: THRESHOLDS[model],
  threshold_source: 'validation_configuration',
});

function mockServer({ loaded = ['custom_cnn', 'resnet18', 'mobilenet_v3_small'], online = true, predictStatus = 200, score = 0.682 } = {}) {
  const calls = [];
  const fetchMock = vi.fn(async (url, init = {}) => {
    calls.push({ url, method: init.method ?? 'GET' });
    if (!online) throw new TypeError('Failed to fetch');
    if (url === '/health') {
      return json(200, { status: 'ready', models: Object.fromEntries(loaded.map((id) => [id, { threshold: THRESHOLDS[id] }])) });
    }
    const model = new URL(url, 'http://test').searchParams.get('model_type');
    if (!loaded.includes(model)) return json(503, { detail: `Model ${model} is not configured` });
    if (predictStatus !== 200) return json(predictStatus, { detail: 'Invalid or undecodable image' });
    const body = predictionBody(model, score);
    if (url.startsWith('/explain')) {
      return json(200, {
        ...body,
        gradcam: {
          target_class: body.prediction,
          target_layer: model === 'resnet18' ? 'layer4.1' : 'features.12',
          score: 'pre-softmax class logit',
          constant_map: false,
          heatmap_shape: [224, 224],
          heatmap_png_base64: 'iVBORw0KGgo=',
        },
      });
    }
    return json(200, body);
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
}

const validImage = async () => ({ ok: true, format: 'PNG', width: 300, height: 300 });
const pngFile = () => new File([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], 'scan.png', { type: 'image/png' });
const modelCalls = (calls) => calls.filter(({ url }) => url !== '/health');

async function setup(options, validate = validImage) {
  const calls = mockServer(options);
  const user = userEvent.setup();
  render(<App validate={validate} />);
  await screen.findAllByText(options?.online === false ? 'API unreachable' : 'API online');
  return { calls, user };
}

async function loadImage(user) {
  await user.upload(screen.getByTestId('file-input'), pngFile());
  await screen.findByAltText(/Uploaded ultrasound image scan.png/);
}

const runButton = () => screen.getByRole('button', { name: /Run analysis/ });

beforeEach(() => {
  URL.createObjectURL = vi.fn(() => 'blob:preview');
  URL.revokeObjectURL = vi.fn();
});
afterEach(() => vi.unstubAllGlobals());

describe('TesticulUS interface', () => {
  it('shows the research disclaimer and API/model status', async () => {
    await setup();
    expect(screen.getAllByText('Research and educational use only. Not for clinical diagnosis.').length).toBeGreaterThan(0);
    expect(screen.getByText(/3 of 3/)).toBeInTheDocument();
    expect(screen.getByText(/Upload only anonymized images you are authorized to share/)).toBeInTheDocument();
  });

  it('previews a selected image without uploading it', async () => {
    const { calls, user } = await setup();
    await loadImage(user);
    expect(screen.getByText(/scan.png · PNG · 300×300 px/)).toBeInTheDocument();
    expect(modelCalls(calls)).toEqual([]);
    expect(runButton()).toBeDisabled();
  });

  it('rejects an invalid file with a readable message', async () => {
    const { calls, user } = await setup({}, async () => ({ ok: false, error: 'Only PNG and JPEG images are supported.' }));
    await user.upload(screen.getByTestId('file-input'), pngFile());
    expect(await screen.findByRole('alert')).toHaveTextContent('Only PNG and JPEG images are supported.');
    expect(screen.queryByAltText(/Uploaded ultrasound image/)).toBeNull();
    expect(modelCalls(calls)).toEqual([]);
  });

  it('marks models whose checkpoints are not loaded as unable to run', async () => {
    await setup({ loaded: ['custom_cnn', 'resnet18'] });
    const mobilenet = screen.getByRole('radio', { name: /MobileNetV3-Small/ });
    expect(mobilenet).toBeDisabled();
    expect(screen.getByText('Checkpoint not loaded — cannot run')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /ResNet18/ })).toBeEnabled();
    expect(screen.getByText(/2 of 3/)).toBeInTheDocument();
  });

  it('disables every model when the API is unreachable', async () => {
    const { user } = await setup({ online: false });
    await loadImage(user);
    for (const radio of screen.getAllByRole('radio')) expect(radio).toBeDisabled();
    expect(runButton()).toBeDisabled();
    expect(screen.getByRole('button', { name: /Retry/ })).toBeInTheDocument();
  });

  it('runs the selected model and renders the server response as a model score', async () => {
    const { calls, user } = await setup();
    await loadImage(user);
    await user.click(screen.getByRole('radio', { name: /ResNet18/ }));
    await user.click(runButton());
    expect(await screen.findByTestId('predicted-class')).toHaveTextContent('Inhomogeneous');
    expect(screen.getByTestId('score')).toHaveTextContent('0.682');
    expect(screen.getByTestId('threshold')).toHaveTextContent('0.3252');
    expect(screen.getByText('Model score for inhomogeneous class')).toBeInTheDocument();
    expect(screen.getByText('not a probability of disease')).toBeInTheDocument();
    expect(screen.getByText(/selected on the validation partition/)).toBeInTheDocument();
    expect(modelCalls(calls)).toEqual([{ url: '/predict?model_type=resnet18', method: 'POST' }]);
  });

  it('reports a model that became unavailable and refreshes availability', async () => {
    const { calls, user } = await setup();
    await loadImage(user);
    await user.click(screen.getByRole('radio', { name: /Custom CNN/ }));
    vi.mocked(fetch).mockImplementationOnce(async () => json(503, { detail: 'Model custom_cnn is not configured' }));
    await user.click(runButton());
    expect(await screen.findByText('This model is not loaded on the server, or the shared research demo is busy. Retry shortly.')).toBeInTheDocument();
    expect(screen.getByText(/Model custom_cnn is not configured/)).toBeInTheDocument();
    expect(screen.queryByTestId('predicted-class')).toBeNull();
    await waitFor(() => expect(calls.filter(({ url }) => url === '/health').length).toBeGreaterThan(1));
  });

  it('shows server-side upload rejections instead of a result', async () => {
    const { user } = await setup({ predictStatus: 400 });
    await loadImage(user);
    await user.click(screen.getByRole('radio', { name: /ResNet18/ }));
    await user.click(runButton());
    expect(await screen.findByText(/could not decode this image/)).toBeInTheDocument();
    expect(screen.queryByTestId('predicted-class')).toBeNull();
  });

  it('flags a result as stale when the model selection changes', async () => {
    const { user } = await setup();
    await loadImage(user);
    await user.click(screen.getByRole('radio', { name: /ResNet18/ }));
    await user.click(runButton());
    await screen.findByTestId('predicted-class');
    await user.click(screen.getByRole('radio', { name: /Custom CNN/ }));
    expect(screen.getByText(/This result belongs to the previous selection/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Generate Grad-CAM/ })).toBeDisabled();
  });

  it('requests Grad-CAM only for a completed result and exposes the overlay toggle', async () => {
    const { calls, user } = await setup();
    await loadImage(user);
    expect(screen.getByRole('button', { name: /Generate Grad-CAM/ })).toBeDisabled();
    await user.click(screen.getByRole('radio', { name: /MobileNetV3-Small/ }));
    await user.click(runButton());
    await screen.findByTestId('predicted-class');
    await user.click(screen.getByRole('button', { name: /Generate Grad-CAM/ }));
    const panel = screen.getByRole('region', { name: /Grad-CAM attribution/ });
    expect(await within(panel).findByText('features.12')).toBeInTheDocument();
    expect(within(panel).getByText(/does not establish clinical correctness/)).toBeInTheDocument();
    const overlay = screen.getByRole('button', { name: /Grad-CAM overlay/ });
    expect(overlay).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', { name: /^Original$/ }));
    expect(overlay).toHaveAttribute('aria-pressed', 'false');
    expect(modelCalls(calls).map(({ url }) => url)).toEqual(['/predict?model_type=mobilenet_v3_small', '/explain?model_type=mobilenet_v3_small']);
  });

  it('explains what the class labels mean and their limits', async () => {
    await setup();
    const about = screen.getByRole('region', { name: /What this prototype is/ });
    expect(within(about).getByRole('heading', { name: 'What the labels mean' })).toBeInTheDocument();
    expect(within(about).getByText(/emerging biomarker for male\s+infertility/)).toBeInTheDocument();
    expect(within(about).getByText(/only still screenshots are kept/)).toBeInTheDocument();
  });

  it('clears the image and its results', async () => {
    const { user } = await setup();
    await loadImage(user);
    await user.click(screen.getByRole('radio', { name: /ResNet18/ }));
    await user.click(runButton());
    await screen.findByTestId('predicted-class');
    await user.click(screen.getByRole('button', { name: 'Remove image' }));
    expect(screen.queryByTestId('predicted-class')).toBeNull();
    expect(screen.getByText('Drop an ultrasound image here')).toBeInTheDocument();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:preview');
  });

  it('labels historical metrics separately from the live prediction and marks MobileNet exploratory', async () => {
    const { user } = await setup();
    const evaluation = screen.getByRole('region', { name: /How the models performed/ });
    expect(within(evaluation).getByText('not the image you upload')).toBeInTheDocument();
    expect(within(evaluation).getAllByText('Exploratory').length).toBeGreaterThan(0);
    await user.click(within(evaluation).getByRole('button', { name: /Table/ }));
    const table = within(evaluation).getByRole('table');
    expect(within(table).getByText('0.7473')).toBeInTheDocument();
    expect(within(table).getByText('10.6 ms')).toBeInTheDocument();
  });
});
