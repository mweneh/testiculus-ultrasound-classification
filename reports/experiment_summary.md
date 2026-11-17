# Frozen experiment summary

Both models completed training, validation threshold selection and final
held-out test evaluation in Google Colab. Experiments are frozen.

Test numbers below were supplied by the project developer, rounded to four
decimal places. Original Colab test artifacts and checkpoints are unavailable
locally and remain unverified. Count-based metrics agree with the supplied
confusion matrices at the reported precision. ROC-AUC and average precision
cannot be reconstructed from confusion matrices.

## Validation: threshold selection

Source: `reports/validation_thresholds.json` (full precision, unchanged).
Maximize positive-class F1 over unique validation probabilities; use `>=`
and choose the highest threshold for exact F1 ties. Test results were not
used for this selection. Validation has 129 images (104 class 0, 25 class 1).

| Model | Frozen threshold | Validation precision | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| custom_cnn | 0.4541136622428894 | 0.3721 | 0.6400 | 0.7404 | 0.4706 |
| resnet18 | 0.3251684010028839 | 0.3261 | 0.6000 | 0.7019 | 0.4225 |

## Test: final frozen evaluation

Both models used the same 129-image held-out set: 104 homogeneous (0),
25 inhomogeneous (positive class 1). Thresholds were fixed from validation.

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| custom_cnn | 0.6123 | 0.4029 | 0.6124 | 0.5317 | 0.4000 | 0.6635 | 0.2857 |
| resnet18 | 0.7473 | 0.5220 | 0.6434 | 0.6421 | 0.6400 | 0.6442 | 0.4103 |

Confusion matrices: rows = actual [0, 1], columns = predicted [0, 1].

- custom_cnn: `[[69, 35], [15, 10]]`.
- resnet18: `[[67, 37], [9, 16]]`.

Majority-class accuracy baseline: 104 / 129 = **80.62%**.
Both model accuracies are below this baseline. An always-homogeneous
classifier has zero positive-class sensitivity and balanced accuracy 0.5.
ResNet18 has higher reported ROC-AUC, average precision, balanced accuracy,
sensitivity and F1; this comparison does not establish statistical significance.

## Reproduction and limits

`python -m src.experiment_summary --output reports/experiment_summary.md`
regenerates this summary from the frozen JSON records, without opening
images, loading models, or reading test manifests. Supply original
`src.evaluation` metrics.json files using `--custom-cnn-metrics` and/or
`--resnet18-metrics` to cross-check them and print their SHA-256 hashes.
Original predictions are needed to independently recompute ROC-AUC/AP;
original checkpoint hashes and runtime versions remain unavailable.

Patient and acquisition-centre identifiers are unavailable. This is a
provisional image-level evaluation, not verified patient-independent or
centre-held-out validation. No clinical validation or diagnostic suitability
is established. Grad-CAM visualizations do not establish localization truth.
