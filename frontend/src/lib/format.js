export const fixed = (value, digits = 3) => (Number.isFinite(value) ? value.toFixed(digits) : '—');

export const percent = (value, digits = 0) => (Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—');

export const milliseconds = (value) => (Number.isFinite(value) ? `${value.toFixed(value < 20 ? 1 : 0)} ms` : '—');

export function compactCount(value) {
  if (!Number.isFinite(value)) return '—';
  if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e7 ? 1 : 2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
  return String(value);
}

export const grouped = (value) => (Number.isFinite(value) ? value.toLocaleString('en-GB') : '—');
