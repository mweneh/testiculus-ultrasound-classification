import { useCallback, useEffect, useRef, useState } from 'react';
import Header from './components/Header.jsx';
import ImageViewer from './components/ImageViewer.jsx';
import ModelSelector from './components/ModelSelector.jsx';
import ResultPanel from './components/ResultPanel.jsx';
import GradCamPanel from './components/GradCamPanel.jsx';
import ComparisonSection from './components/ComparisonSection.jsx';
import AboutSection from './components/AboutSection.jsx';
import { useHealth } from './hooks/useHealth.js';
import { ApiError, explain as requestExplanation, predict as requestPrediction } from './lib/api.js';
import { MODEL_BY_ID } from './lib/models.js';
import { validateImageFile } from './lib/upload.js';

const IDLE = { phase: 'idle' };
let imageCounter = 0;

export default function App({ validate = validateImageFile }) {
  const health = useHealth();
  const [image, setImage] = useState(null);
  const [imageError, setImageError] = useState(null);
  const [validating, setValidating] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [analysis, setAnalysis] = useState(IDLE);
  const [explanation, setExplanation] = useState(IDLE);
  const [overlay, setOverlay] = useState({ visible: true, opacity: 0.45 });
  const requestRef = useRef(null);

  const abortRequests = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
  }, []);

  // Release the preview URL when the image changes or the app unmounts.
  useEffect(() => () => image && URL.revokeObjectURL(image.url), [image]);
  useEffect(() => abortRequests, [abortRequests]);

  // A model that stops being available cannot stay selected.
  useEffect(() => {
    if (selectedModel && health.phase === 'online' && !health.models[selectedModel]) setSelectedModel(null);
  }, [health.phase, health.models, selectedModel]);

  const resetOutputs = () => {
    abortRequests();
    setAnalysis(IDLE);
    setExplanation(IDLE);
  };

  const handleFile = async (file) => {
    setImageError(null);
    setValidating(true);
    const check = await validate(file);
    setValidating(false);
    if (!check.ok) {
      setImageError(check.error);
      return;
    }
    resetOutputs();
    imageCounter += 1;
    setImage({
      key: imageCounter,
      file,
      url: URL.createObjectURL(file),
      name: file.name || 'upload',
      size: file.size,
      format: check.format,
      width: check.width,
      height: check.height,
    });
  };

  const clearImage = () => {
    resetOutputs();
    setImage(null);
    setImageError(null);
  };

  const modelReady = Boolean(selectedModel && health.models[selectedModel]);
  const blockers = [];
  if (!image) blockers.push('load an image');
  if (health.phase === 'offline') blockers.push('reconnect to the API');
  else if (!selectedModel) blockers.push('choose an available model');
  const canRun = Boolean(image && modelReady && health.phase === 'online');

  const runAnalysis = async () => {
    if (!image) return setImageError('Load a PNG or JPEG image before running the analysis.');
    if (!modelReady) {
      setAnalysis({ phase: 'error', error: new ApiError('The selected model is not available on the server.', { status: 503 }) });
      return undefined;
    }
    abortRequests();
    const controller = new AbortController();
    requestRef.current = controller;
    const context = { imageKey: image.key, model: selectedModel };
    setExplanation(IDLE);
    setAnalysis({ phase: 'running', ...context });
    const started = performance.now();
    try {
      const result = await requestPrediction(selectedModel, image.file, { signal: controller.signal });
      setAnalysis({ phase: 'success', result, completedAt: new Date(), roundTripMs: performance.now() - started, ...context });
    } catch (error) {
      if (error?.name === 'AbortError') return undefined;
      if (error?.status === 503) health.refresh();
      setAnalysis({ phase: 'error', error, ...context });
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
    }
    return undefined;
  };

  const analysisCurrent = analysis.phase === 'success' && analysis.imageKey === image?.key && analysis.model === selectedModel;
  const stale = analysis.phase === 'success' && !analysisCurrent;

  const runExplanation = async () => {
    if (!analysisCurrent || !modelReady) return;
    abortRequests();
    const controller = new AbortController();
    requestRef.current = controller;
    const context = { imageKey: image.key, model: selectedModel };
    setExplanation({ phase: 'running', ...context });
    try {
      const result = await requestExplanation(selectedModel, image.file, { signal: controller.signal });
      setExplanation({ phase: 'success', result, ...context });
      setOverlay((current) => ({ ...current, visible: true }));
    } catch (error) {
      if (error?.name === 'AbortError') return;
      if (error?.status === 503) health.refresh();
      setExplanation({ phase: 'error', error, ...context });
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
    }
  };

  const explanationCurrent =
    explanation.phase !== 'idle' && explanation.imageKey === image?.key && explanation.model === selectedModel;
  const shownExplanation = explanationCurrent ? explanation : IDLE;
  const gradcamBlocked = !analysisCurrent
    ? 'Run the analysis for the current image and model first.'
    : !modelReady
      ? 'The selected model is not available.'
      : null;
  const runningModel = analysis.phase === 'running' ? MODEL_BY_ID[analysis.model]?.name : MODEL_BY_ID[selectedModel]?.name;

  return (
    <div id="top" className="min-h-screen">
      <a href="#analyse" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-accent focus:px-4 focus:py-2 focus:text-accent-ink">
        Skip to analysis
      </a>
      <Header health={health} />

      <main className="mx-auto max-w-7xl px-4 pb-20 sm:px-6 lg:px-8">
        <section id="analyse" aria-labelledby="analyse-title" className="scroll-mt-28 pt-6 sm:pt-8">
          <div className="max-w-3xl">
            <p className="eyebrow">Image-level classification · frozen research models</p>
            <h1 id="analyse-title" className="mt-2 text-[26px] font-semibold leading-tight tracking-tight text-ink-1 sm:text-[32px]">
              Classify testicular ultrasound appearance
            </h1>
            <p className="mt-2 text-[15px] text-ink-2">
              Homogeneous or inhomogeneous, decided by each model&apos;s frozen validation-selected threshold. Results describe this
              image only.
            </p>
          </div>

          <div className="mt-6 grid gap-5 lg:grid-cols-12">
            <div className="min-w-0 lg:col-span-7 xl:col-span-8">
              <div className="lg:sticky lg:top-32">
                <ImageViewer
                  image={image}
                  validating={validating}
                  error={imageError}
                  onFile={handleFile}
                  onClear={clearImage}
                  analysing={analysis.phase === 'running' || explanation.phase === 'running'}
                  analysingModelName={explanation.phase === 'running' ? `Grad-CAM for ${runningModel}` : runningModel}
                  explanation={shownExplanation.phase === 'success' ? shownExplanation.result : null}
                  overlay={overlay}
                  onOverlayChange={setOverlay}
                />
              </div>
            </div>
            <div className="min-w-0 space-y-5 lg:col-span-5 xl:col-span-4">
              <ModelSelector
                health={health}
                selected={selectedModel}
                onSelect={setSelectedModel}
                disabled={analysis.phase === 'running'}
              />
              <ResultPanel
                analysis={analysis}
                blockers={blockers}
                stale={stale}
                onRun={runAnalysis}
                onRetry={runAnalysis}
                canRun={canRun}
                running={analysis.phase === 'running'}
                modelName={runningModel}
              />
              <GradCamPanel
                explanation={shownExplanation}
                available={Boolean(!gradcamBlocked && analysisCurrent)}
                onExplain={runExplanation}
                blockedReason={gradcamBlocked}
              />
            </div>
          </div>
        </section>

        <div className="my-16 h-px bg-lab-line sm:my-20" />
        <ComparisonSection />
        <div className="my-16 h-px bg-lab-line sm:my-20" />
        <AboutSection />
      </main>

      <footer className="border-t border-lab-line">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-[12.5px] text-ink-3 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
          <p>TesticulUS research prototype</p>
          <p className="font-medium text-caution">Research and educational use only. Not for clinical diagnosis.</p>
        </div>
      </footer>
    </div>
  );
}
