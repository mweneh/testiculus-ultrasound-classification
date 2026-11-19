import { useState } from 'react';
import { ChartBar, Table2 } from 'lucide-react';
import { MODEL_IDS } from '../lib/api.js';
import { comparison, METRICS, METRIC_LABELS, observedStrengths } from '../lib/records.js';
import { MODELS } from '../lib/models.js';
import { compactCount, fixed, grouped, milliseconds } from '../lib/format.js';
import { ExploratoryBadge, Note, OriginalBadge, SectionHeading, SeriesSwatch } from './ui.jsx';

const BAR_FILL = { 1: 'bg-series-1', 2: 'bg-series-2', 3: 'bg-series-3' };

function Legend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-5 gap-y-2" aria-label="Model legend">
      {MODELS.map((model) => (
        <li key={model.id} className="flex items-center gap-2 text-[13px] text-ink-2">
          <SeriesSwatch series={model.series} />
          {model.name}
          {model.exploratory && <span className="text-caution">*</span>}
        </li>
      ))}
      <li className="text-[12px] text-ink-3">
        <span className="text-caution">*</span> post-hoc exploratory
      </li>
    </ul>
  );
}

function Tooltip({ value, label, detail }) {
  return (
    <span
      role="tooltip"
      className="pointer-events-none absolute -top-2 left-1/2 z-10 hidden -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-lg border border-lab-strong bg-lab-page px-2.5 py-1.5 text-left shadow-panel group-hover:block group-focus-visible:block"
    >
      <span className="block font-mono text-[13px] font-semibold text-ink-1">{value}</span>
      <span className="block text-[11.5px] text-ink-2">{label}</span>
      {detail && <span className="block text-[11.5px] text-ink-3">{detail}</span>}
    </span>
  );
}

function MetricPanel({ metric }) {
  const reference = metric.reference;
  return (
    <figure className="panel p-4" aria-labelledby={`metric-${metric.key}`}>
      <figcaption>
        <p id={`metric-${metric.key}`} className="text-sm font-semibold text-ink-1">
          {metric.label}
        </p>
        <p className="mt-0.5 min-h-[2.5rem] text-[12px] leading-snug text-ink-3">{metric.description}</p>
      </figcaption>
      <div className="relative mt-3 space-y-2.5">
        {reference && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute -bottom-1 -top-1 z-0 w-px bg-ink-3/50"
            style={{ left: `calc(8rem + (100% - 8rem - 3.25rem) * ${reference.value})` }}
          />
        )}
        {MODELS.map((model) => {
          const record = comparison.models[model.id];
          const value = record.metrics[metric.key];
          const detail = metric.detail?.(record.confusion);
          return (
            <div
              key={model.id}
              tabIndex={0}
              aria-label={`${model.name}: ${metric.label} ${fixed(value, 4)}${detail ? `, ${detail}` : ''}`}
              className="group relative flex items-center gap-2 rounded-md outline-offset-2"
            >
              <span className="w-[7.5rem] shrink-0 truncate text-[12px] text-ink-2">{model.name}</span>
              <span className="relative h-3.5 flex-1">
                <span
                  className={`absolute inset-y-0 left-0 rounded-r-[4px] ${BAR_FILL[model.series]} transition-[filter] group-hover:brightness-125 group-focus-visible:brightness-125`}
                  style={{ width: `${value * 100}%` }}
                />
                <Tooltip value={fixed(value, 4)} label={model.name} detail={detail} />
              </span>
              <span className="tabular w-[2.75rem] shrink-0 text-right font-mono text-[12px] text-ink-1">{fixed(value, 3)}</span>
            </div>
          );
        })}
      </div>
      <div className="mt-2 flex justify-between pl-[8rem] pr-[3.25rem] font-mono text-[10px] text-ink-3" aria-hidden="true">
        <span>0</span>
        {reference ? <span>{reference.label} {fixed(reference.value, 2)}</span> : <span />}
        <span>1</span>
      </div>
    </figure>
  );
}

function LatencyPanel() {
  const { benchmark } = comparison;
  const maxP95 = Math.max(...MODEL_IDS.map((id) => comparison.models[id].latency.p95));
  const axisMax = Math.ceil(maxP95 / 25) * 25;
  const ticks = Array.from({ length: axisMax / 25 + 1 }, (_, index) => index * 25);
  return (
    <figure className="panel p-4 sm:col-span-2 lg:col-span-3" aria-labelledby="latency-title">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-2">
        <p id="latency-title" className="text-sm font-semibold text-ink-1">
          Median CPU forward latency (ms) · whisker marks the 95th percentile
        </p>
        <p className="text-[12px] text-ink-3">Lower is faster</p>
      </figcaption>
      <div className="mt-4 space-y-3">
        {MODELS.map((model) => {
          const { latency, checkpointMiB, parameters } = comparison.models[model.id];
          return (
            <div
              key={model.id}
              tabIndex={0}
              aria-label={`${model.name}: median ${milliseconds(latency.median)}, 95th percentile ${milliseconds(latency.p95)}`}
              className="group relative flex items-center gap-2 rounded-md"
            >
              <span className="w-[8rem] shrink-0 truncate text-[12px] text-ink-2">{model.name}</span>
              <span className="relative h-3.5 flex-1">
                <span
                  className={`absolute inset-y-0 left-0 rounded-r-[4px] ${BAR_FILL[model.series]} group-hover:brightness-125 group-focus-visible:brightness-125`}
                  style={{ width: `${(latency.median / axisMax) * 100}%` }}
                />
                <span
                  aria-hidden="true"
                  className="absolute top-1/2 h-px -translate-y-1/2 bg-ink-2"
                  style={{ left: `${(latency.median / axisMax) * 100}%`, width: `${((latency.p95 - latency.median) / axisMax) * 100}%` }}
                />
                <span aria-hidden="true" className="absolute -top-0.5 -bottom-0.5 w-px bg-ink-2" style={{ left: `${(latency.p95 / axisMax) * 100}%` }} />
                <Tooltip
                  value={`${milliseconds(latency.median)} median`}
                  label={model.name}
                  detail={`p95 ${milliseconds(latency.p95)} · ${compactCount(parameters)} params · ${checkpointMiB.toFixed(1)} MiB`}
                />
              </span>
              <span className="tabular w-[4.5rem] shrink-0 text-right font-mono text-[12px] text-ink-1">{milliseconds(latency.median)}</span>
            </div>
          );
        })}
      </div>
      <div className="relative mt-2 ml-[8.5rem] mr-[5rem] h-4 font-mono text-[10px] text-ink-3" aria-hidden="true">
        {ticks.map((tick) => (
          <span key={tick} className="absolute -translate-x-1/2" style={{ left: `${(tick / axisMax) * 100}%` }}>
            {tick}
          </span>
        ))}
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-3">
        Model forward only: batch {benchmark.batchSize}, {benchmark.inputSize.join('×')} input, {benchmark.dtype}, {benchmark.threads} threads
        on {benchmark.cpu.replace(/\(R\)|\(TM\)/g, '')}; {benchmark.warmup} warm-up and {benchmark.repeats} timed forwards per model.
        Excludes decoding, preprocessing and API overhead.
      </p>
    </figure>
  );
}

function SummaryCards() {
  const strengths = observedStrengths();
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {MODELS.map((model) => {
        const record = comparison.models[model.id];
        return (
          <article key={model.id} className="panel flex flex-col p-4">
            <div className="flex items-center justify-between gap-2">
              <h3 className="flex items-center gap-2 text-[15px] font-semibold text-ink-1">
                <SeriesSwatch series={model.series} />
                {model.name}
              </h3>
              {model.exploratory ? <ExploratoryBadge /> : <OriginalBadge />}
            </div>
            <p className="mt-2 text-[12.5px] leading-snug text-ink-2">{model.summary}</p>
            <dl className="mt-4 grid grid-cols-3 gap-2 border-y border-lab-line py-3 text-center">
              <div>
                <dt className="text-[11px] text-ink-3">ROC-AUC</dt>
                <dd className="mt-0.5 font-mono text-[15px] text-ink-1">{fixed(record.metrics.roc_auc, 3)}</dd>
              </div>
              <div>
                <dt className="text-[11px] text-ink-3">Detected</dt>
                <dd className="mt-0.5 font-mono text-[15px] text-ink-1">
                  {record.confusion.tp}/{record.confusion.tp + record.confusion.fn}
                </dd>
              </div>
              <div>
                <dt className="text-[11px] text-ink-3">Median CPU</dt>
                <dd className="mt-0.5 font-mono text-[15px] text-ink-1">{milliseconds(record.latency.median)}</dd>
              </div>
            </dl>
            <p className="eyebrow mt-3">Observed on this test partition</p>
            <ul className="mt-2 flex flex-wrap gap-1.5">
              {strengths[model.id].length ? (
                strengths[model.id].map((item) => (
                  <li key={item} className="rounded-md bg-lab-raised px-2 py-1 text-[11.5px] text-ink-2">
                    {item}
                  </li>
                ))
              ) : (
                <li className="text-[12px] text-ink-3">No metric where it was highest</li>
              )}
            </ul>
            <p className="mt-auto pt-3 text-[11.5px] text-ink-3">
              {record.confusion.fp} false positives · {record.confusion.fn} missed inhomogeneous images
            </p>
          </article>
        );
      })}
    </div>
  );
}

function ComparisonTable() {
  const rows = [
    ...['roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy', 'precision', 'recall_sensitivity', 'specificity', 'f1'].map(
      (key) => [METRIC_LABELS[key], (record) => fixed(record.metrics[key], 4)],
    ),
    ['Frozen threshold', (record) => fixed(record.threshold, 4)],
    ['TP / FP / TN / FN', ({ confusion: c }) => `${c.tp} / ${c.fp} / ${c.tn} / ${c.fn}`],
    ['Median latency', (record) => milliseconds(record.latency.median)],
    ['95th percentile latency', (record) => milliseconds(record.latency.p95)],
    ['Parameters (trainable)', (record) => `${grouped(record.parameters)} (${grouped(record.trainableParameters)})`],
    ['Checkpoint size', (record) => `${record.checkpointMiB.toFixed(2)} MiB`],
  ];
  return (
    <div className="panel overflow-x-auto">
      <table className="w-full min-w-[560px] text-left text-[13px]">
        <caption className="sr-only">Frozen test-partition results and CPU efficiency for each model</caption>
        <thead>
          <tr className="border-b border-lab-line text-ink-2">
            <th scope="col" className="px-4 py-3 font-medium">Measure</th>
            {MODELS.map((model) => (
              <th key={model.id} scope="col" className="px-4 py-3 text-right font-medium">
                <span className="inline-flex items-center gap-2">
                  <SeriesSwatch series={model.series} />
                  {model.name}
                  {model.exploratory && <span className="text-caution">*</span>}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label} className="border-b border-lab-line/60 last:border-0">
              <th scope="row" className="px-4 py-2.5 font-normal text-ink-2">
                {label}
              </th>
              {MODEL_IDS.map((id) => (
                <td key={id} className="tabular px-4 py-2.5 text-right font-mono text-ink-1">
                  {value(comparison.models[id])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ComparisonSection() {
  const [view, setView] = useState('charts');
  const { test } = comparison;
  return (
    <section id="evaluation" aria-labelledby="evaluation-title" className="scroll-mt-28">
      <SectionHeading id="evaluation-title" eyebrow="Frozen evaluation record · historical results" title="How the models performed in the completed experiment">
        <p>
          These figures come from the saved test-partition evaluation: {test.total} images, {test.positive} of them inhomogeneous.
          They describe the frozen models, <strong className="font-semibold text-ink-1">not the image you upload</strong>. A
          majority-class rule would reach {fixed(test.majorityBaseline, 3)} accuracy, so accuracy alone is not shown in the charts.
        </p>
      </SectionHeading>

      <div className="mt-5 flex flex-wrap gap-2">
        <span className="chip">Single training seed per model</span>
        <span className="chip">No confidence intervals · descriptive only</span>
        <span className="chip">Image-level split</span>
        <span className="chip">One sensitivity step = 1 of {test.positive} images</span>
      </div>

      <Note tone="warning" className="mt-5 max-w-4xl">
        MobileNetV3-Small was added after the original Custom CNN and ResNet18 test results had been observed. Its checkpoint and
        threshold were frozen before its single test evaluation, but the comparison including it is{' '}
        <strong className="font-semibold text-ink-1">exploratory, not confirmatory</strong>. No model is shown to be clinically
        superior; the trade-offs depend on an unmeasured clinical cost of missed versus false findings.
      </Note>

      <div className="mt-8">
        <SummaryCards />
      </div>

      <div className="mt-10 flex flex-wrap items-center justify-between gap-4">
        <Legend />
        <div role="group" aria-label="Display" className="inline-flex rounded-lg border border-lab-line bg-lab-panel p-0.5">
          {[
            ['charts', 'Charts', ChartBar],
            ['table', 'Table', Table2],
          ].map(([value, label, Icon]) => (
            <button
              key={value}
              type="button"
              aria-pressed={view === value}
              onClick={() => setView(value)}
              className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold ${
                view === value ? 'bg-lab-raised text-ink-1' : 'text-ink-3 hover:text-ink-1'
              }`}
            >
              <Icon aria-hidden="true" className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4">
        {view === 'charts' ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {METRICS.map((metric) => (
              <MetricPanel key={metric.key} metric={metric} />
            ))}
            <LatencyPanel />
          </div>
        ) : (
          <ComparisonTable />
        )}
      </div>
      <p className="mt-3 text-[12px] text-ink-3">
        Source: reports/mobilenet_exploratory_results.json (frozen record; all three models evaluated on identical test paths and labels).
        Thresholds are model-specific and validation-selected.
      </p>
    </section>
  );
}
