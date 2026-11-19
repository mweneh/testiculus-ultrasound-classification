import { describe, expect, it } from 'vitest';
import { models as raw } from '@records/mobilenet_exploratory_results.json';
import { comparison, observedStrengths } from './records.js';
import { MODELS } from './models.js';

describe('frozen evaluation record', () => {
  it('passes saved metrics through unchanged', () => {
    for (const id of ['custom_cnn', 'resnet18', 'mobilenet_v3_small']) {
      const source = raw[id].test.metrics;
      const derived = comparison.models[id];
      expect(derived.metrics.roc_auc).toBe(source.roc_auc);
      expect(derived.metrics.average_precision).toBe(source.average_precision);
      expect(derived.metrics.recall_sensitivity).toBe(source.recall_sensitivity);
      expect(derived.metrics.f1).toBe(source.f1);
      expect(derived.threshold).toBe(source.threshold);
      expect(derived.latency.median).toBe(raw[id].efficiency.latency.median_ms);
    }
  });

  it('matches the values reported in the scientific report', () => {
    const at = (id, key, digits = 4) => comparison.models[id].metrics[key].toFixed(digits);
    expect([at('custom_cnn', 'roc_auc'), at('resnet18', 'roc_auc'), at('mobilenet_v3_small', 'roc_auc')]).toEqual(['0.6123', '0.7473', '0.6562']);
    expect([at('custom_cnn', 'average_precision'), at('resnet18', 'average_precision'), at('mobilenet_v3_small', 'average_precision')]).toEqual(['0.4029', '0.5220', '0.4889']);
    expect([at('custom_cnn', 'recall_sensitivity', 2), at('resnet18', 'recall_sensitivity', 2), at('mobilenet_v3_small', 'recall_sensitivity', 2)]).toEqual(['0.40', '0.64', '0.36']);
    expect([at('custom_cnn', 'f1'), at('resnet18', 'f1'), at('mobilenet_v3_small', 'f1')]).toEqual(['0.2857', '0.4103', '0.4390']);
    expect(['custom_cnn', 'resnet18', 'mobilenet_v3_small'].map((id) => comparison.models[id].latency.median.toFixed(2))).toEqual(['51.47', '62.17', '10.59']);
    expect(comparison.test).toMatchObject({ total: 129, positive: 25, negative: 104 });
    expect(comparison.dataset).toEqual({ total: 860, negative: 696, positive: 164 });
  });

  it('keeps MobileNetV3-Small marked as post-hoc exploratory', () => {
    expect(MODELS.map((model) => [model.id, model.exploratory])).toEqual([
      ['custom_cnn', false],
      ['resnet18', false],
      ['mobilenet_v3_small', true],
    ]);
    expect(comparison.models.mobilenet_v3_small.status).toBe('post-hoc exploratory');
  });

  it('derives observed strengths from the record', () => {
    const strengths = observedStrengths();
    expect(strengths.resnet18).toEqual(expect.arrayContaining(['Highest ROC-AUC', 'Highest average precision', 'Highest sensitivity']));
    expect(strengths.mobilenet_v3_small).toEqual(expect.arrayContaining(['Highest F1 score', 'Fastest median CPU forward']));
    expect(strengths.custom_cnn).toEqual(['Fewest parameters']);
  });
});
