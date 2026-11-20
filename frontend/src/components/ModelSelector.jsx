import { CircleCheck, CircleSlash } from 'lucide-react';
import { MODELS } from '../lib/models.js';
import { compactCount } from '../lib/format.js';
import { ExploratoryBadge, SeriesSwatch } from './ui.jsx';

function availability(health, id) {
  if (health.phase === 'checking') return { ready: false, label: 'Checking availability…' };
  if (health.phase === 'offline') return { ready: false, label: 'API unreachable' };
  const threshold = health.models[id]?.threshold;
  if (threshold === undefined) return { ready: false, label: 'Checkpoint not loaded — cannot run' };
  return { ready: true, label: `Ready · threshold ${threshold.toFixed(4)}`, threshold };
}

export default function ModelSelector({ health, selected, onSelect, disabled }) {
  return (
    <fieldset className="panel p-4" disabled={disabled}>
      <legend className="sr-only">Choose a model</legend>
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <p className="text-sm font-semibold text-ink-1" aria-hidden="true">
          1 · Choose a model
        </p>
        <span className="text-[11px] text-ink-3">Frozen checkpoints</span>
      </div>
      <div className="space-y-2">
        {MODELS.map((model) => {
          const state = availability(health, model.id);
          const checked = selected === model.id;
          return (
            <label
              key={model.id}
              className={`group relative flex cursor-pointer gap-3 rounded-xl border px-3 py-2.5 transition-colors duration-150 focus-within:ring-2 focus-within:ring-accent ${
                checked ? 'border-accent/60 bg-accent/[0.06]' : 'border-lab-line hover:border-lab-strong hover:bg-lab-raised/50'
              } ${state.ready ? '' : 'cursor-not-allowed opacity-60'}`}
            >
              <input
                type="radio"
                name="model"
                value={model.id}
                checked={checked}
                disabled={!state.ready}
                onChange={() => onSelect(model.id)}
                className="sr-only"
                aria-describedby={`${model.id}-state`}
              />
              <span
                aria-hidden="true"
                className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${
                  checked ? 'border-accent' : 'border-lab-strong'
                }`}
              >
                {checked && <span className="h-2 w-2 rounded-full bg-accent" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <SeriesSwatch series={model.series} />
                  <span className="text-sm font-semibold text-ink-1">{model.name}</span>
                  {model.exploratory && <ExploratoryBadge />}
                </span>
                <span className="mt-1 flex flex-wrap items-center justify-between gap-x-3 gap-y-0.5 text-[12px] leading-snug">
                  <span className="text-ink-2">
                    {model.training} · {compactCount(model.parameters)} params
                  </span>
                  <span id={`${model.id}-state`} className={`inline-flex items-center gap-1 font-mono text-[11px] ${state.ready ? 'text-accent-soft' : 'text-ink-3'}`}>
                    {state.ready ? (
                      <CircleCheck aria-hidden="true" className="h-3.5 w-3.5" />
                    ) : (
                      <CircleSlash aria-hidden="true" className="h-3.5 w-3.5" />
                    )}
                    {state.label}
                  </span>
                </span>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
