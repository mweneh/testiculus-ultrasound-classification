import { Info, TriangleAlert } from 'lucide-react';

export function SeriesSwatch({ series, className = '' }) {
  const colors = { 1: 'bg-series-1', 2: 'bg-series-2', 3: 'bg-series-3' };
  return <span aria-hidden="true" className={`inline-block h-2.5 w-2.5 shrink-0 rounded-[3px] ${colors[series]} ${className}`} />;
}

export function ExploratoryBadge() {
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full border border-caution/30 bg-caution/10 px-2 py-0.5 text-[11px] font-semibold text-caution"
      title="Exploratory, not confirmatory."
    >
      Exploratory
    </span>
  );
}

export function Note({ children, tone = 'info', className = '' }) {
  const Icon = tone === 'warning' ? TriangleAlert : Info;
  const tones = {
    info: 'border-lab-line bg-lab-raised/50 text-ink-2',
    warning: 'border-caution/30 bg-caution/5 text-ink-2',
  };
  return (
    <div className={`flex gap-2.5 rounded-xl border px-3.5 py-3 text-[13px] leading-relaxed ${tones[tone]} ${className}`}>
      <Icon aria-hidden="true" className={`mt-0.5 h-4 w-4 shrink-0 ${tone === 'warning' ? 'text-caution' : 'text-accent'}`} />
      <div>{children}</div>
    </div>
  );
}

export function ErrorMessage({ error, onRetry, retryLabel = 'Try again' }) {
  if (!error) return null;
  return (
    <div role="alert" className="flex gap-2.5 rounded-xl border border-danger/30 bg-danger/5 px-3.5 py-3 text-[13px] text-ink-1">
      <TriangleAlert aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-danger" />
      <div className="min-w-0 flex-1">
        <p className="font-medium">{error.message}</p>
        {error.detail && error.detail !== error.message && (
          <p className="mt-0.5 break-words text-ink-2">Server detail: {error.detail}</p>
        )}
        {onRetry && (
          <button type="button" onClick={onRetry} className="mt-2 text-[13px] font-semibold text-accent-soft underline-offset-2 hover:underline">
            {retryLabel}
          </button>
        )}
      </div>
    </div>
  );
}

export function SectionHeading({ id, eyebrow, title, children }) {
  return (
    <div className="max-w-3xl">
      <p className="eyebrow">{eyebrow}</p>
      <h2 id={id} className="mt-2 text-2xl font-semibold tracking-tight text-ink-1 sm:text-[28px]">
        {title}
      </h2>
      {children && <div className="mt-3 text-[15px] leading-relaxed text-ink-2">{children}</div>}
    </div>
  );
}
