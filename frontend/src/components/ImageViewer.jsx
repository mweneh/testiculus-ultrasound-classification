import { useRef, useState } from 'react';
import { Eye, EyeOff, ImagePlus, LoaderCircle, RefreshCw, ScanLine, Upload, X } from 'lucide-react';
import { formatBytes } from '../lib/upload.js';
import HeatmapLayer, { HEAT_RGB } from './HeatmapLayer.jsx';

const ACCEPT = 'image/png,image/jpeg,.png,.jpg,.jpeg';

function Dropzone({ onFile, validating, inputRef }) {
  const [dragging, setDragging] = useState(false);
  const depth = useRef(0);

  return (
    <div
      onDragEnter={(event) => {
        event.preventDefault();
        depth.current += 1;
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => {
        depth.current = Math.max(0, depth.current - 1);
        if (depth.current === 0) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        depth.current = 0;
        setDragging(false);
        const file = event.dataTransfer?.files?.[0];
        if (file) onFile(file);
      }}
      data-testid="dropzone"
      className={`relative flex h-full min-h-[300px] flex-col items-center justify-center rounded-xl border border-dashed px-6 py-10 text-center transition-colors duration-200 ${
        dragging ? 'border-accent bg-accent/[0.06]' : 'border-lab-strong bg-lab-stage/60'
      }`}
    >
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-lab-line bg-lab-raised">
        {validating ? (
          <LoaderCircle aria-hidden="true" className="h-6 w-6 animate-spin text-accent" />
        ) : (
          <ImagePlus aria-hidden="true" className="h-6 w-6 text-accent" />
        )}
      </div>
      <p className="mt-5 text-lg font-semibold text-ink-1">
        {validating ? 'Checking image…' : dragging ? 'Release to load the image' : 'Drop an ultrasound image here'}
      </p>
      <p className="mt-1.5 max-w-sm text-sm text-ink-2">PNG or JPEG, up to 5 MiB and 20 megapixels.</p>
      <button type="button" className="btn-ghost mt-5" onClick={() => inputRef.current?.click()} disabled={validating}>
        <Upload aria-hidden="true" className="h-4 w-4" />
        Choose a file
      </button>
      <p className="mt-6 max-w-md text-xs leading-relaxed text-ink-3">
        The image stays in your browser until you press <span className="font-semibold text-ink-2">Run analysis</span>.
        The server processes it in memory and does not store it.
      </p>
    </div>
  );
}

function OverlayControls({ explanation, overlay, onOverlayChange }) {
  const constant = explanation.gradcam.constantMap;
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-3 border-t border-lab-line px-4 py-3">
      <div role="group" aria-label="Viewer layer" className="inline-flex rounded-lg border border-lab-line bg-lab-stage p-0.5">
        {[
          [false, 'Original', EyeOff],
          [true, 'Grad-CAM overlay', Eye],
        ].map(([value, label, Icon]) => (
          <button
            key={label}
            type="button"
            aria-pressed={overlay.visible === value}
            onClick={() => onOverlayChange({ ...overlay, visible: value })}
            className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
              overlay.visible === value ? 'bg-lab-raised text-ink-1' : 'text-ink-3 hover:text-ink-1'
            }`}
          >
            <Icon aria-hidden="true" className="h-3.5 w-3.5" />
            {label}
          </button>
        ))}
      </div>
      <label className="flex items-center gap-2.5 text-xs text-ink-2">
        <span>Overlay strength</span>
        <input
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={overlay.opacity}
          disabled={!overlay.visible}
          onChange={(event) => onOverlayChange({ ...overlay, opacity: Number(event.target.value) })}
          className="h-1.5 w-28 cursor-pointer accent-accent disabled:cursor-not-allowed disabled:opacity-40"
        />
        <span className="tabular w-9 font-mono text-ink-3">{Math.round(overlay.opacity * 100)}%</span>
      </label>
      <div className="flex items-center gap-2 text-[11px] text-ink-3" aria-label="Attribution colour scale">
        <span>Low</span>
        <span
          aria-hidden="true"
          className="h-2 w-20 rounded-full border border-lab-line"
          style={{ background: `linear-gradient(90deg, rgba(${HEAT_RGB.join(',')}, 0), rgba(${HEAT_RGB.join(',')}, 1))` }}
        />
        <span>High relative attribution</span>
      </div>
      {constant && <p className="w-full text-xs text-caution">This map is constant for this image, so the overlay shows no spatial structure.</p>}
    </div>
  );
}

export default function ImageViewer({ image, validating, error, onFile, onClear, analysing, analysingModelName, explanation, overlay, onOverlayChange }) {
  const inputRef = useRef(null);
  const ratio = image ? image.width / image.height : 1;

  return (
    <section aria-labelledby="viewer-title" className="panel flex flex-col overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-lab-line px-4 py-3">
        <div className="min-w-0">
          <h2 id="viewer-title" className="flex items-center gap-2 text-sm font-semibold text-ink-1">
            <ScanLine aria-hidden="true" className="h-4 w-4 text-accent" />
            Ultrasound image
          </h2>
          {image ? (
            <p className="mt-0.5 truncate font-mono text-[11px] text-ink-3" title={image.name}>
              {image.name} · {image.format} · {image.width}×{image.height} px · {formatBytes(image.size)}
            </p>
          ) : (
            <p className="mt-0.5 text-[12px] text-ink-3">No image loaded</p>
          )}
        </div>
        {image && (
          <div className="flex gap-2">
            <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={() => inputRef.current?.click()} disabled={analysing}>
              <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
              Replace
            </button>
            <button type="button" className="btn-ghost px-3 py-1.5 text-xs" onClick={onClear} aria-label="Remove image">
              <X aria-hidden="true" className="h-3.5 w-3.5" />
              Remove
            </button>
          </div>
        )}
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="sr-only"
        tabIndex={-1}
        aria-label="Upload ultrasound image"
        data-testid="file-input"
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = '';
          if (file) onFile(file);
        }}
      />

      <div className="p-4 [--stage-h:300px] sm:[--stage-h:420px] lg:[--stage-h:500px]">
        {image ? (
          <div className="relative flex items-center justify-center rounded-xl bg-lab-stage" style={{ height: 'calc(var(--stage-h) + 2rem)' }}>
            <div
              className="relative overflow-hidden rounded-md ring-1 ring-lab-line"
              style={{ aspectRatio: `${image.width} / ${image.height}`, width: `min(calc(100% - 2rem), calc(var(--stage-h) * ${ratio}))` }}
            >
              <img src={image.url} alt={`Uploaded ultrasound image ${image.name}`} className="absolute inset-0 h-full w-full select-none" draggable="false" />
              {explanation && (
                <HeatmapLayer pngBase64={explanation.gradcam.pngBase64} opacity={overlay.opacity} visible={overlay.visible} />
              )}
              {analysing && (
                <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden motion-reduce:hidden">
                  <div className="h-1/3 w-full animate-scan bg-gradient-to-b from-transparent via-accent/15 to-transparent" />
                </div>
              )}
            </div>
            {analysing && (
              <p className="absolute bottom-3 left-1/2 inline-flex -translate-x-1/2 items-center gap-2 rounded-full border border-lab-line bg-lab-page/85 px-3 py-1 text-xs text-ink-2">
                <LoaderCircle aria-hidden="true" className="h-3.5 w-3.5 animate-spin text-accent" />
                Running {analysingModelName}…
              </p>
            )}
          </div>
        ) : (
          <div style={{ height: 'calc(var(--stage-h) + 2rem)' }}>
            <Dropzone onFile={onFile} validating={validating} inputRef={inputRef} />
          </div>
        )}
        {error && (
          <p role="alert" className="mt-3 rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-sm text-ink-1">
            {error}
          </p>
        )}
      </div>

      {image && explanation && <OverlayControls explanation={explanation} overlay={overlay} onOverlayChange={onOverlayChange} />}
    </section>
  );
}
