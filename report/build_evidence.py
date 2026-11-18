"""Audit saved numeric artifacts and render report tables/figures; never run models.

--collect requires locally supplied CSVs and writes only report/ derivatives.
--verify-local checks those CSVs against the committed report snapshot.
Without either flag, rendering needs only the tracked snapshot and matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
NAMES = {'custom_cnn': 'Custom CNN', 'resnet18': 'ResNet18',
         'mobilenet_v3_small': 'MobileNetV3-Small'}
METRICS = ['roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy',
           'precision', 'recall_sensitivity', 'specificity', 'f1']
LABELS = ['ROC-AUC', 'Average Precision', 'Accuracy', 'Balanced accuracy',
          'Precision', 'Sensitivity', 'Specificity', 'F1-score']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def numeric_metrics(rows, threshold):
    """Independent count verification plus standard ranking metrics on saved scores."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = [int(r['label']) for r in rows]
    p = [float(r['probability_class_1']) for r in rows]
    assert all(v in (0, 1) for v in y)
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in p)
    cm = [[0, 0], [0, 0]]
    for label, prob in zip(y, p):
        cm[label][int(prob >= threshold)] += 1
    tn, fp, fn, tp = *cm[0], *cm[1]
    return dict(roc_auc=float(roc_auc_score(y, p)),
                average_precision=float(average_precision_score(y, p)),
                accuracy=(tn + tp) / len(y),
                balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall_sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                f1=2 * tp / (2 * tp + fp + fn), confusion_matrix=cm)


def identities(rows):
    indexed = {r['image_path']: int(r['label']) for r in rows}
    assert len(indexed) == len(rows), 'Duplicate path in saved numeric record'
    return indexed


def assert_metrics(actual, expected):
    for key in METRICS:
        assert math.isclose(actual[key], expected[key], abs_tol=1e-12), key
    cm = expected['confusion_matrix']
    if isinstance(cm, dict):
        cm = cm['rows_actual_columns_predicted']
    assert actual['confusion_matrix'] == cm


def collect():
    """Read existing records only. No image loaders, checkpoints, training or inference."""
    original = read_json(ROOT / 'reports/artifact_verification.json')
    aggregate = read_json(ROOT / 'reports/mobilenet_exploratory_results.json')
    efficiency = read_json(ROOT / 'reports/model_efficiency.json')
    configs = [read_json(ROOT / 'reports/validation_thresholds.json'),
               read_json(ROOT / 'reports/mobilenet_v3_small_thresholds.json')]
    configured = {k: v for c in configs for k, v in c['models'].items()}
    split_rows = {n: read_csv(ROOT / f'data/splits/{n}.csv')
                  for n in ['train', 'validation', 'test']}
    ids = {n: identities(rows) for n, rows in split_rows.items()}
    assert not (ids['train'].keys() & ids['validation'].keys())
    assert not (ids['train'].keys() & ids['test'].keys())
    assert not (ids['validation'].keys() & ids['test'].keys())
    counts = {n: [list(d.values()).count(i) for i in [0, 1]] for n, d in ids.items()}
    assert counts == {'train': [488, 114], 'validation': [104, 25], 'test': [104, 25]}
    sources = ['data/splits/train.csv', 'data/splits/validation.csv', 'data/splits/test.csv',
               'reports/artifact_verification.json', 'reports/validation_thresholds.json',
               'reports/mobilenet_exploratory_results.json', 'reports/model_efficiency.json',
               'reports/mobilenet_v3_small_freeze.json', 'reports/mobilenet_v3_small_thresholds.json',
               'reports/mobilenet_application_verification.json',
               'reports/mobilenet_container_verification.json', 'reports/similarity_audit.md']
    models = {}
    for name in NAMES:
        config = configured[name]
        threshold = config['selected_threshold']
        val_path = ROOT / f'results/{name}_validation/predictions.csv'
        assert digest(val_path) == config['predictions_sha256']
        val_rows = read_csv(val_path)
        assert identities(val_rows) == ids['validation']
        val = numeric_metrics(val_rows, threshold)
        default_val = numeric_metrics(val_rows, .5)
        sweep = config['threshold_sweep']
        assert len(sweep) == len(set(float(r['probability_class_1']) for r in val_rows)) == 129
        for point in sweep:
            measured = numeric_metrics(val_rows, point['threshold'])
            for key in ['precision', 'specificity', 'f1']:
                assert math.isclose(point[key], measured[key], abs_tol=1e-12)
            assert math.isclose(point['recall'], measured['recall_sensitivity'], abs_tol=1e-12)
        assert max(sweep, key=lambda p: (p['f1'], p['threshold']))['threshold'] == threshold
        for key, value in config['metrics_at_selected_threshold'].items():
            assert math.isclose(val[key], value, abs_tol=1e-12), (name, key)
        test_path = ROOT / f'results/{name}_test/predictions.csv'
        test_rows = read_csv(test_path)
        assert identities(test_rows) == ids['test']
        recorded_test = aggregate['models'][name]['test']
        assert digest(test_path) == recorded_test['predictions_sha256']
        test = numeric_metrics(test_rows, threshold)
        assert_metrics(test, recorded_test['metrics'])
        metrics_path = ROOT / f'results/{name}_test/metrics.json'
        metric_file = read_json(metrics_path)
        assert_metrics(test, metric_file.get('metrics', metric_file))
        training = (aggregate['freeze']['training'] if name == 'mobilenet_v3_small'
                    else original['models'][name]['checkpoint_metadata'])
        history_path = ROOT / f'training_runs/{name}/history.csv'
        history = [{k: int(v) if k == 'epoch' else float(v) for k, v in r.items()}
                   for r in read_csv(history_path)]
        assert len(history) == (10 if name == 'mobilenet_v3_small' else original['models'][name]['epochs_completed'])
        best = min(history, key=lambda r: r['validation_loss'])
        assert best['epoch'] == training['best_epoch']
        assert math.isclose(best['validation_loss'], training['best_validation_loss'], abs_tol=1e-12)
        models[name] = dict(threshold=threshold, validation=val, validation_at_half=default_val,
                            test=test, training=training, history=history,
                            efficiency=efficiency['models'][name])
        sources.extend([str(p.relative_to(ROOT)) for p in [val_path, test_path, metrics_path, history_path]])
    return dict(format_version=1, split_counts=counts, models=models,
                source_sha256={p: digest(ROOT / p) for p in sources},
                safeguards={'model_execution': False, 'images_opened': False,
                            'experimental_records_modified': False})


def render_tables(data):
    models = data['models']
    tables = {}
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Metric & Custom CNN & ResNet18 & MobileNetV3-Small$^{*}$ \\', r'\midrule']
    for key, label in zip(METRICS, LABELS):
        lines.append(label + ' & ' + ' & '.join(f"{m['test'][key]:.4f}" for m in models.values()) + r' \\')
    tables['test_metrics.tex'] = '\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n'
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Model & Partition & AUC & AP & Sens. & Spec. & F1 \\', r'\midrule']
    for name, m in models.items():
        for part in ['validation', 'test']:
            vals = ' & '.join(f"{m[part][k]:.4f}" for k in ['roc_auc', 'average_precision', 'recall_sensitivity', 'specificity', 'f1'])
            lines.append(f"{NAMES[name]} & {part.capitalize()} & {vals}" + r' \\')
    tables['validation_test.tex'] = '\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n'
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Model & Parameters & Checkpoint & Median & Mean & 95th pct. \\',
             r' & & (MiB) & (ms) & (ms) & (ms) \\', r'\midrule']
    for name, m in models.items():
        e = m['efficiency']; latency = e['latency']
        lines.append(f"{NAMES[name]} & {e['parameters']:,} & {e['checkpoint_bytes']/1048576:.3f} & {latency['median_ms']:.2f} & {latency['mean_ms']:.2f} & {latency['p95_ms']:.2f}" + r' \\')
    tables['efficiency.tex'] = '\n'.join(lines + [r'\bottomrule', r'\end{tabular}']) + '\n'
    return tables


def render_figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    metadata = {'CreationDate': None, 'ModDate': None}
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.4), layout='constrained')
    for ax, (name, model) in zip(axes, data['models'].items()):
        cm = np.array(model['test']['confusion_matrix'])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                fraction = cm[i, j] / cm[i].sum()
                ax.text(j, i, f'{cm[i,j]}\n({fraction:.1%})', ha='center', va='center',
                        color='white' if fraction > .55 else '#122336')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Homogeneous', 'Inhomogeneous'],
               yticklabels=['Homogeneous', 'Inhomogeneous'], xlabel='Predicted class',
               title=NAMES[name] + ('*' if name == 'mobilenet_v3_small' else ''))
    axes[0].set_ylabel('Actual class')
    fig.savefig(OUT / 'figures/confusion_matrices.pdf', metadata=metadata)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.0), layout='constrained', sharey=True)
    for ax, (name, model) in zip(axes, data['models'].items()):
        h = model['history']; epochs = [r['epoch'] for r in h]
        ax.plot(epochs, [r['train_loss'] for r in h], label='Training', color='#007a79')
        ax.plot(epochs, [r['validation_loss'] for r in h], label='Validation', color='#9b3848')
        ax.axvline(model['training']['best_epoch'], color='#737373', ls=':', label='Best checkpoint')
        ax.set(xlabel='Epoch', title=NAMES[name] + ('*' if name == 'mobilenet_v3_small' else ''), ylim=(0, 1.85))
    axes[0].set_ylabel('Recorded weighted loss')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / 'figures/training_histories.pdf', metadata=metadata)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--collect', action='store_true')
    group.add_argument('--verify-local', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check tables without rewriting figures')
    args = parser.parse_args()
    path = OUT / 'evidence.json'
    if args.collect:
        data = collect()
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    else:
        data = read_json(path)
    if args.verify_local:
        assert collect() == data, 'Report snapshot differs from source artifacts'
    for name, content in render_tables(data).items():
        path = OUT / 'generated' / name
        if args.check:
            assert path.read_text() == content, f'Stale report table: {name}'
        else:
            path.write_text(content)
    if not args.check:
        render_figures(data)
    print('Report evidence/tables verified; no models executed or images opened.')


if __name__ == '__main__':
    main()
