import { CircleCheck, Clock, Info, Play, TriangleAlert } from 'lucide-react';
import { MODEL_BY_ID } from '../lib/models.js';
import { ErrorMessage, ExploratoryBadge, SeriesSwatch } from './ui.jsx';

const CLASS_LABEL = { homogeneous: 'Homogeneous', inhomogeneous: 'Inhomogeneous' };

function ScoreMeter({ score, threshold }) {
  const scorePct = `${(score * 100).toFixed(2)}%`;
  const thresholdPct = `${(threshold * 100).toFixed(2)}%`;
  return (
    <div className="mt-5">
      <div className="relative h-10">
        <div className="absolute inset-x-0 top-5 h-2.5 overflow-hidden rounded-full bg-lab-stage ring-1 ring-lab-line">
          <div className="absolute inset-y-0 right-0 bg-ink-3/15" style={{ left: thresholdPct }} aria-hidden="true" />
          <div className="relative h-full rounded-full bg-accent transition-[width] duration-700 ease-out" style={{ width: scorePct }} />
        </div>
        <div className="absolute top-0 flex -translate-x-1/2 flex-col items-center" style={{ left: thresholdPct }} aria-hidden="true">
          <span className="whitespace-nowrap font-mono text-[10.5px] text-ink-2">threshold</span>
          <span className="mt-0.5 h-6 w-0.5 rounded-full bg-ink-1" />
        </div>
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10.5px] text-ink-3" aria-hidden="true">
        <span>0 · homogeneous</span>
        <span>inhomogeneous · 1</span>
      </div>
    </div>
  );
}

function Placeholder({ blockers }) {
  return (
    <div className="flex flex-col items-center rounded-xl border border-dashed border-lab-line px-5 py-8 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-lab-raised">
        <Play aria-hidden="true" className="h-4 w-4 text-ink-3" />
      </div>
      <p className="mt-3 text-sm font-medium text-ink-1">No analysis yet</p>
      <p className="mt-1 max-w-xs text-[13px] text-ink-2">
        {blockers.length ? `To begin: ${blockers.join(' and ')}.` : 'Press Run analysis to send this image to the selected model.'}
      </p>
    </div>
  );
}

function LoadingState({ modelName }) {
  return (
    <div aria-busy="true" className="space-y-3">
      <p className="sr-only">Running {modelName}</p>
      <div className="skeleton h-4 w-28" />
      <div className="skeleton h-9 w-48" />
      <div className="skeleton h-3 w-full" />
      <div className="skeleton h-3 w-2/3" />
    </div>
  );
}

export function ResultBody({ result, stale, completedAt, roundTripMs }) {
  const model = MODEL_BY_ID[result.modelType];
  const above = result.score >= result.threshold;
  return (
    <div className={`transition-opacity ${stale ? 'opacity-55' : ''}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-0.5 text-[11.5px] font-semibold text-accent-soft">
          <CircleCheck aria-hidden="true" className="h-3.5 w-3.5" />
          Completed
        </span>
        <span className="inline-flex items-center gap-1.5 text-[12px] text-ink-2">
          <SeriesSwatch series={model.series} />
          {model.name}
        </span>
        {model.exploratory && <ExploratoryBadge />}
      </div>

      <p className="eyebrow mt-5">Predicted image label</p>
      <p className="mt-1 text-[34px] font-semibold leading-tight tracking-tight text-ink-1" data-testid="predicted-class">
        {CLASS_LABEL[result.className]}
      </p>
      <p className="mt-0.5 text-[13px] text-ink-2">
        Class {result.prediction} · score {above ? 'at or above' : 'below'} the decision threshold
      </p>

      <ScoreMeter score={result.score} threshold={result.threshold} />

      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-xl border border-lab-line bg-lab-stage/50 p-3">
          <dt className="text-[12px] text-ink-3">Model score for inhomogeneous class</dt>
          <dd className="mt-1 font-mono text-xl font-medium text-ink-1" data-testid="score" title={String(result.score)}>
            {result.score.toFixed(3)}
          </dd>
        </div>
        <div className="rounded-xl border border-lab-line bg-lab-stage/50 p-3">
          <dt className="text-[12px] text-ink-3">Decision threshold</dt>
          <dd className="mt-1 font-mono text-xl font-medium text-ink-1" data-testid="threshold" title={String(result.threshold)}>
            {result.threshold.toFixed(4)}
          </dd>
        </div>
      </dl>

      <p className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-ink-3">
        <span className="inline-flex items-center gap-1">
          <Clock aria-hidden="true" className="h-3 w-3" />
          {completedAt.toLocaleTimeString('en-GB')}
        </span>
        <span>Round trip {Math.round(roundTripMs)} ms (upload + API + model)</span>
        <span>Threshold source: validation configuration</span>
      </p>

      <div className="mt-4 flex gap-2.5 rounded-xl border border-lab-line bg-lab-raised/50 px-3.5 py-3 text-[12.5px] leading-relaxed text-ink-2">
        <Info aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
        <p>
          The score is the model&apos;s softmax output for the inhomogeneous class. It is not calibrated and is{' '}
          <strong className="font-semibold text-ink-1">not a probability of disease</strong>. Each model&apos;s threshold was
          selected on the validation partition (maximum F1 for the inhomogeneous class) and frozen before test evaluation.
        </p>
      </div>
    </div>
  );
}

export default function ResultPanel({ analysis, blockers, stale, onRun, canRun, running, modelName, onRetry }) {
  return (
    <section aria-labelledby="result-title" className="panel p-4">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 id="result-title" className="text-sm font-semibold text-ink-1">
          2 · Run analysis
        </h2>
        <span className="text-[11px] text-ink-3">Live result for the current image</span>
      </div>
      <button type="button" className="btn-primary w-full py-3 text-[15px]" onClick={onRun} disabled={!canRun || running}>
        <Play aria-hidden="true" className="h-4 w-4" />
        {running ? `Running ${modelName}…` : 'Run analysis'}
      </button>
      {!canRun && !running && blockers.length > 0 && (
        <p className="mt-2 text-center text-[12px] text-ink-3">Requires: {blockers.join(' · ')}</p>
      )}

      <div className="mt-4" aria-live="polite">
        {analysis.phase === 'idle' && <Placeholder blockers={blockers} />}
        {analysis.phase === 'running' && <LoadingState modelName={modelName} />}
        {analysis.phase === 'error' && <ErrorMessage error={analysis.error} onRetry={canRun ? onRetry : undefined} />}
        {analysis.phase === 'success' && (
          <>
            {stale && (
              <p className="mb-3 flex items-center gap-2 rounded-lg border border-caution/30 bg-caution/5 px-3 py-2 text-[12.5px] text-ink-1">
                <TriangleAlert aria-hidden="true" className="h-3.5 w-3.5 shrink-0 text-caution" />
                The model selection changed. This result belongs to the previous selection; run the analysis again to update it.
              </p>
            )}
            <ResultBody result={analysis.result} stale={stale} completedAt={analysis.completedAt} roundTripMs={analysis.roundTripMs} />
          </>
        )}
      </div>
    </section>
  );
}
