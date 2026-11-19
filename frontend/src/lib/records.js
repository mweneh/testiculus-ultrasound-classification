/**
 * Historical evaluation figures, derived only from tracked frozen records:
 *   reports/mobilenet_exploratory_results.json  (test metrics, latency, status)
 *   report/evidence.json                        (split counts)
 * The files are bundled at build time. Nothing here re-evaluates a model.
 */
import {
  benchmark_conditions as benchmarkConditions,
  experiment_status as extensionStatus,
  hardware,
  identical_test_paths_and_labels as identicalTestPaths,
  models as recordModels,
} from '@records/mobilenet_exploratory_results.json';
import { split_counts as splitCounts } from '@report/evidence.json';
import { MODEL_IDS } from './api.js';

const MIB = 1024 * 1024;

function modelRecord(id) {
  const entry = recordModels[id];
  const metrics = entry.test.metrics;
  const matrix = metrics.confusion_matrix;
  return {
    id,
    status: entry.comparison_status,
    exploratory: entry.comparison_status !== 'original frozen comparison',
    threshold: metrics.threshold,
    metrics: {
      roc_auc: metrics.roc_auc,
      average_precision: metrics.average_precision,
      accuracy: metrics.accuracy,
      balanced_accuracy: metrics.balanced_accuracy,
      precision: metrics.precision,
      recall_sensitivity: metrics.recall_sensitivity,
      specificity: metrics.specificity,
      f1: metrics.f1,
    },
    confusion: { tp: matrix.tp, fp: matrix.fp, tn: matrix.tn, fn: matrix.fn },
    majorityBaseline: metrics.majority_class_baseline_accuracy,
    nSamples: entry.test.n_samples,
    classCounts: { negative: entry.test.class_counts['0'], positive: entry.test.class_counts['1'] },
    parameters: entry.efficiency.parameters,
    trainableParameters: entry.efficiency.trainable_parameters,
    checkpointMiB: entry.efficiency.checkpoint_bytes / MIB,
    latency: {
      median: entry.efficiency.latency.median_ms,
      mean: entry.efficiency.latency.mean_ms,
      p95: entry.efficiency.latency.p95_ms,
      min: entry.efficiency.latency.min_ms,
      repeats: entry.efficiency.latency.measurements_ms.length,
    },
  };
}

const models = Object.fromEntries(MODEL_IDS.map((id) => [id, modelRecord(id)]));
const reference = models.resnet18;

function sumSplit([negative, positive]) {
  return { negative, positive, total: negative + positive };
}

const splits = {
  train: sumSplit(splitCounts.train),
  validation: sumSplit(splitCounts.validation),
  test: sumSplit(splitCounts.test),
};

export const comparison = {
  models,
  extensionStatus,
  identicalTestPaths,
  test: {
    total: reference.nSamples,
    positive: reference.classCounts.positive,
    negative: reference.classCounts.negative,
    prevalence: reference.classCounts.positive / reference.nSamples,
    majorityBaseline: reference.majorityBaseline,
  },
  benchmark: {
    cpu: hardware.cpu_model,
    threads: hardware.intraop_threads,
    torch: hardware.torch_version,
    batchSize: benchmarkConditions.batch_size,
    inputSize: benchmarkConditions.input_size,
    warmup: benchmarkConditions.warmup,
    repeats: benchmarkConditions.repeats,
    dtype: benchmarkConditions.dtype,
    scope: benchmarkConditions.scope,
  },
  splits,
  dataset: {
    total: splits.train.total + splits.validation.total + splits.test.total,
    negative: splits.train.negative + splits.validation.negative + splits.test.negative,
    positive: splits.train.positive + splits.validation.positive + splits.test.positive,
  },
};

/** Threshold-free and frozen-threshold metrics shown as small multiples. */
export const METRICS = [
  {
    key: 'roc_auc',
    phrase: 'ROC-AUC',
    label: 'ROC-AUC',
    description: 'How well scores rank inhomogeneous above homogeneous images. Threshold-free.',
    reference: { value: 0.5, label: 'Chance' },
  },
  {
    key: 'average_precision',
    phrase: 'average precision',
    label: 'Average precision',
    description: 'Precision summarized across recall levels. Threshold-free.',
    reference: { value: comparison.test.prevalence, label: 'Prevalence' },
  },
  {
    key: 'recall_sensitivity',
    phrase: 'sensitivity',
    label: 'Sensitivity',
    description: 'Inhomogeneous test images detected at the frozen threshold.',
    detail: (c) => `${c.tp} of ${c.tp + c.fn} inhomogeneous images detected`,
  },
  {
    key: 'specificity',
    phrase: 'specificity',
    label: 'Specificity',
    description: 'Homogeneous test images correctly left unflagged.',
    detail: (c) => `${c.tn} of ${c.tn + c.fp} homogeneous images left unflagged`,
  },
  {
    key: 'precision',
    phrase: 'precision',
    label: 'Precision',
    description: 'Inhomogeneous calls that were correct.',
    detail: (c) => `${c.tp} of ${c.tp + c.fp} inhomogeneous calls correct`,
  },
  {
    key: 'f1',
    phrase: 'F1 score',
    label: 'F1 score',
    description: 'Balance of precision and sensitivity for the inhomogeneous class.',
  },
];

const EXTRA_LABELS = { accuracy: 'Accuracy', balanced_accuracy: 'Balanced accuracy' };
export const METRIC_LABELS = {
  ...Object.fromEntries(METRICS.map(({ key, label }) => [key, label])),
  ...EXTRA_LABELS,
};

/** Models attaining the highest observed value; ties keep every leader. */
function leaders(values, prefer = 'max') {
  const best = prefer === 'max' ? Math.max(...values.map(([, v]) => v)) : Math.min(...values.map(([, v]) => v));
  return values.filter(([, value]) => value === best).map(([id]) => id);
}

/** Observed per-model strengths, computed from the record (not significance-tested). */
export function observedStrengths() {
  const strengths = Object.fromEntries(MODEL_IDS.map((id) => [id, []]));
  for (const { key, phrase } of METRICS) {
    for (const id of leaders(MODEL_IDS.map((model) => [model, models[model].metrics[key]]))) {
      strengths[id].push(`Highest ${phrase}`);
    }
  }
  for (const id of leaders(MODEL_IDS.map((model) => [model, models[model].latency.median]), 'min')) {
    strengths[id].push('Fastest median CPU forward');
  }
  for (const id of leaders(MODEL_IDS.map((model) => [model, models[model].parameters]), 'min')) {
    strengths[id].push('Fewest parameters');
  }
  return strengths;
}
