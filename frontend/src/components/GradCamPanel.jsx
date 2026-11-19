import { Layers, LoaderCircle } from 'lucide-react';
import { ErrorMessage, Note } from './ui.jsx';

const CLASS_LABEL = { 0: 'Homogeneous (class 0)', 1: 'Inhomogeneous (class 1)' };

export default function GradCamPanel({ explanation, available, onExplain, blockedReason }) {
  const running = explanation.phase === 'running';
  const ready = explanation.phase === 'success' ? explanation.result : null;
  return (
    <section aria-labelledby="gradcam-title" className="panel p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 id="gradcam-title" className="flex items-center gap-2 text-sm font-semibold text-ink-1">
            <Layers aria-hidden="true" className="h-4 w-4 text-accent" />
            3 · Grad-CAM attribution
          </h2>
          <p className="mt-1 text-[12.5px] text-ink-2">Optional coarse map of where the predicted class score is locally sensitive.</p>
        </div>
        <span className="chip shrink-0 text-[11px]">Optional</span>
      </div>

      <div className="mt-3" aria-live="polite">
        {ready ? (
          <dl className="grid grid-cols-1 gap-x-4 gap-y-2 rounded-xl border border-lab-line bg-lab-stage/50 p-3 text-[12.5px] sm:grid-cols-2">
            <div>
              <dt className="text-ink-3">Explained class</dt>
              <dd className="text-ink-1">{CLASS_LABEL[ready.gradcam.targetClass]}</dd>
            </div>
            <div>
              <dt className="text-ink-3">Target layer</dt>
              <dd className="font-mono text-ink-1">{ready.gradcam.targetLayer}</dd>
            </div>
            <div>
              <dt className="text-ink-3">Differentiated score</dt>
              <dd className="text-ink-1">{ready.gradcam.score}</dd>
            </div>
            <div>
              <dt className="text-ink-3">Normalization</dt>
              <dd className="text-ink-1">Min–max within this image</dd>
            </div>
          </dl>
        ) : (
          <button
            type="button"
            className="btn-ghost w-full"
            onClick={onExplain}
            disabled={!available || running}
            title={available ? undefined : blockedReason}
          >
            {running ? <LoaderCircle aria-hidden="true" className="h-4 w-4 animate-spin" /> : <Layers aria-hidden="true" className="h-4 w-4" />}
            {running ? 'Computing Grad-CAM…' : 'Generate Grad-CAM for this result'}
          </button>
        )}
        {!ready && !available && blockedReason && <p className="mt-2 text-center text-[12px] text-ink-3">{blockedReason}</p>}
        {explanation.phase === 'error' && (
          <div className="mt-3">
            <ErrorMessage error={explanation.error} onRetry={available ? onExplain : undefined} />
          </div>
        )}
      </div>

      <Note className="mt-3">
        Grad-CAM is a visualization of model sensitivity at a coarse feature-map resolution, upsampled to the image. It is not
        a segmentation or lesion localization, and a plausible-looking map does not establish clinical correctness. Overlays of
        TesticulUS images are adaptations under CC BY-NC-SA 4.0.
      </Note>
    </section>
  );
}
