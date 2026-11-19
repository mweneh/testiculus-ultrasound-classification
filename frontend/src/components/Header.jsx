import { Cpu, RotateCw, ShieldAlert } from 'lucide-react';
import { MODEL_IDS } from '../lib/api.js';

function Logo() {
  return (
    <svg viewBox="0 0 32 32" aria-hidden="true" className="h-9 w-9 shrink-0">
      <rect width="32" height="32" rx="9" fill="#13213a" />
      <path d="M6 21a10 10 0 0 1 20 0" fill="none" stroke="#2dd4bf" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M10.5 21a5.5 5.5 0 0 1 11 0" fill="none" stroke="#5eead4" strokeWidth="2.2" strokeLinecap="round" opacity=".7" />
      <circle cx="16" cy="21" r="1.8" fill="#e8eef6" />
    </svg>
  );
}

function ApiStatus({ health }) {
  const config = {
    checking: { dot: 'bg-ink-3 animate-pulse', label: 'Checking API…' },
    online: { dot: 'bg-accent', label: 'API online' },
    offline: { dot: 'bg-danger', label: 'API unreachable' },
  }[health.phase];
  const ready = Object.keys(health.models).length;
  return (
    <div className="flex flex-wrap items-center gap-2" role="status" aria-live="polite">
      <span className="chip">
        <span aria-hidden="true" className={`h-2 w-2 rounded-full ${config.dot}`} />
        {config.label}
      </span>
      {health.phase === 'online' && (
        <span className="chip" title="Models whose trained checkpoints the server loaded at startup">
          <Cpu aria-hidden="true" className="h-3.5 w-3.5 text-ink-3" />
          <span>
            {ready} of {MODEL_IDS.length}
            <span className="hidden sm:inline"> models</span> ready
          </span>
        </span>
      )}
      {health.phase === 'offline' && (
        <button type="button" onClick={health.refresh} className="chip hover:border-lab-strong hover:text-ink-1">
          <RotateCw aria-hidden="true" className="h-3.5 w-3.5" />
          Retry
        </button>
      )}
    </div>
  );
}

export default function Header({ health }) {
  return (
    <header className="z-30 border-b md:sticky md:top-0 border-lab-line bg-lab-page/85 backdrop-blur supports-[backdrop-filter]:bg-lab-page/70">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-x-6 gap-y-2.5 px-4 py-3 sm:px-6 lg:px-8">
        <a href="#top" className="flex items-center gap-3 rounded-lg">
          <Logo />
          <span className="leading-tight">
            <span className="block text-[17px] font-semibold tracking-tight text-ink-1">TesticulUS</span>
            <span className="block text-xs text-ink-3">Ultrasound Intelligence Lab</span>
          </span>
          <span className="ml-1 hidden rounded-full border border-accent/30 bg-accent/10 px-2 py-0.5 text-[11px] font-semibold text-accent-soft sm:inline">
            Research prototype
          </span>
        </a>
        <nav aria-label="Sections" className="order-3 hidden w-full gap-1 text-sm text-ink-2 md:order-none md:flex md:w-auto">
          {[
            ['#analyse', 'Analyse'],
            ['#evaluation', 'Evaluation record'],
            ['#about', 'About the research'],
          ].map(([href, label]) => (
            <a key={href} href={href} className="rounded-lg px-3 py-1.5 hover:bg-lab-raised hover:text-ink-1">
              {label}
            </a>
          ))}
        </nav>
        <ApiStatus health={health} />
      </div>
      <div className="border-t border-caution/20 bg-caution/[0.06]">
        <p className="mx-auto flex max-w-7xl items-center gap-2 px-4 py-1.5 text-[12.5px] font-medium text-caution sm:px-6 lg:px-8">
          <ShieldAlert aria-hidden="true" className="h-3.5 w-3.5 shrink-0" />
          Research and educational use only. Not for clinical diagnosis.
        </p>
      </div>
    </header>
  );
}
