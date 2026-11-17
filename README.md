# TesticulUS Ultrasound Classification

Reproducible binary image-classification pipeline for testicular ultrasound
images. The repository provides a custom CNN and an ImageNet-initialized
ResNet18, fixed image-level data splits, checkpointed training, and
checkpoint-based evaluation and single-image inference.

Actual validation-based threshold-selection results from Google Colab are
preserved in [reports/validation_thresholds.json](reports/validation_thresholds.json).
These are validation results; no held-out test results are reported here.
The threshold integration did not retrain models or access the held-out test
partition.

## Dataset and labels

The supervised dataset uses grayscale PNG files under `data/raw/`:

| Directory | Label | Images |
|---|---:|---:|
| `homogeneous/` | 0 | 696 |
| `inhomogeneous/` | 1 (positive) | 164 |
| `imgs/` | none | 200 |

Only the first two directories enter the classification manifests. The 200
unlabelled images and `masks/` files are excluded from supervised classification;
masks are not model inputs. The labelled images are 300×300 pixels. Retain
`homogeneous/793.png` for full-image classification even though its
segmentation mask is empty. Exclude it only from work that requires a valid
segmentation mask.

### Provenance and limitations

The local directory structure is labelled TesticulUS, but this checkout has no
archive checksum, release identifier, provider receipt, or accompanying license
file that establishes the exact source version. The local copy contains 1,060
images across `homogeneous/`, `inhomogeneous/`, and `imgs/` (860 labelled and
200 unlabelled). The [official dataset page](https://ditto.ing.unimore.it/testiculus/)
now describes TesticulUS-Real as 1,053 real images and also describes the
separate synthetic TesticulUS-Syn release. The local copy therefore is not
treated as verified byte-for-byte identical to that public release. Confirm
the exact provenance and reuse terms with the data provider before
redistribution.

Related publications and the authors' resources:

- [Morelli et al., “Enhancing Testicular Ultrasound Image Classification Through
  Synthetic Data and Pretraining Strategies” (ICIAP 2025)](https://federicobolelli.it/pub_files/2025iciap.pdf)
- [Morelli et al., “A New Multicenter Testicular US Dataset and a Lightweight
  Cond-UNet for Generalization in US Segmentation” (BMVC 2026)](https://iris.unimore.it/handle/11380/1416328)
- [Authors' TesticulUS repository](https://github.com/AImageLab-zip/TesticulUS)

Those sources describe research datasets, but this repository does not contain
verified mappings from the local image paths to patient/examination or clinical
centre IDs. Do not claim patient-independent, centre-held-out, or cross-centre
validation from the local image-level split. The author inquiry about metadata
is pending.

## Split and similarity audit

The checked-in manifests in `data/splits/` were generated using seed 42 and a
stratified 70/15/15 image-level split. They contain class-qualified paths, so
same-stem files from different class directories remain distinct. The counts
are 602 train (488 class 0, 114 class 1), 129 validation (104/25), and 129 test
(104/25). `reports/similarity_audit.md` records these counts and the audit.

The audit found no exact duplicate pixel arrays. A 64-bit grayscale difference
hash (dHash) with Hamming distance ≤4 flagged 22 candidate pairs, including
pairs in different partitions. These are ambiguous review flags, not proof of
duplicate acquisition or shared patient identity. Human review is pending; do
not claim those pairs identify the same patient. Review them and document any
grouping decision before training. The audit does not establish patient or
centre independence.

Regenerate the manifests and audit from the repository root:

```bash
python -m src.data.splits \
  --data-dir data/raw \
  --output-dir data/splits \
  --seed 42 \
  --near-threshold 4 \
  --report-path reports/similarity_audit.md
```

The splitter keeps exact duplicate groups together and fails if identical pixel
arrays have conflicting class labels. Perceptual-hash candidates are reported
for review and are not automatically grouped. Split manifests contain paths
and labels, not medical image pixels.

Check the existing manifests and render all flagged pairs without regenerating
or modifying the manifests:

```bash
python -m src.data.similarity_review \
  --data-dir data/raw \
  --split-dir data/splits \
  --audit reports/similarity_audit.md \
  --output reports/similarity_contact_sheet.png \
  --expected-total 860
```

The command verifies complete and unique manifest coverage, class labels,
resolvable image paths, per-class split proportions (within 1.5 percentage
points of the 70/15/15 targets), report counts, and recomputed perceptual-hash
findings. It writes the contact sheet under `reports/`; image IDs in the sheet
are filename stems, not patient identifiers.

## Setup

Python 3.10 or later is recommended. Install dependencies in a virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In Google Colab, install the same dependencies with:

```bash
python -m pip install -r requirements.txt
```

Place the labelled images under `data/raw/homogeneous/` and
`data/raw/inhomogeneous/`, with the optional unlabelled images and masks in
their documented folders. Do not commit raw images, virtual environments, or
model checkpoints.

## Train a model

Training uses only `train.csv` and `validation.csv`. Class weights are computed
from training labels only. Training augmentations are stochastic; validation
preprocessing is deterministic. Early stopping and learning-rate scheduling
use validation loss. The selected checkpoint and epoch history are written to
the output directory.

Custom CNN:

```bash
python -m src.training \
  --model custom_cnn \
  --data-dir data/raw \
  --split-dir data/splits \
  --output-dir checkpoints/custom_cnn \
  --epochs 50 \
  --patience 7 \
  --seed 42
```

ImageNet-pretrained ResNet18 (downloads weights on first use if not cached):

```bash
python -m src.training \
  --model resnet18 \
  --pretrained \
  --data-dir data/raw \
  --split-dir data/splits \
  --output-dir checkpoints/resnet18 \
  --epochs 50 \
  --patience 7 \
  --seed 42
```

Training defaults to CUDA when available and otherwise uses CPU. It supports a
fixed seed, AdamW, weighted cross-entropy, `ReduceLROnPlateau`, and early
stopping. A checkpoint stores the model state, architecture, class mapping,
preprocessing details, initialization flag, seed, class weights, and best
validation loss. Tests use synthetic tensors and do not download pretrained
weights or train on the medical images.

## Evaluate a selected checkpoint

Evaluation is a separate command and requires an explicit manifest. It loads
the frozen validation-selected threshold for the checkpoint's `model_type`
from `reports/validation_thresholds.json` by default. No threshold value needs
to be copied into code or shell commands. An alternate configuration can be
supplied with `--threshold-config`.

The following final-evaluation example is for use only after checkpoint and
threshold selection are complete. It was not run during this integration:

```bash
python -m src.evaluation \
  --checkpoint checkpoints/custom_cnn/best_model.pth \
  --data-dir data/raw \
  --manifest data/splits/test.csv \
  --output-dir artifacts/evaluation/custom_cnn_test \
  --threshold-config reports/validation_thresholds.json
```

For ResNet18, use its checkpoint and a separate output directory; the loader
automatically selects the `resnet18` record. Never tune thresholds or select a
model using test results. The explicit `--threshold 0.5` option remains
available for baseline comparisons; it cannot be combined with
`--threshold-config`. Final evaluation should use the frozen configuration.

Evaluation writes `metrics.json`, a per-image `predictions.csv`, and ROC and
precision-recall plots. The positive class is inhomogeneous (1). Report ROC-AUC,
average precision (the precision-recall summary), accuracy, balanced accuracy,
precision, recall/sensitivity, specificity, F1, and the confusion matrix.
Metrics whose required labels or predictions are absent are represented as
`null` and named in `undefined_metrics`. Reports identify whether the threshold
came from the validation configuration or an explicit override, and prediction
rows include the applied threshold.

### Select thresholds from validation predictions

The frozen Colab experiment used the 129-image validation partition (104
homogeneous, 25 inhomogeneous). For each selected checkpoint, the procedure was:

1. Collect deterministic validation predictions with class 1 as the positive
   class. Verify that class-qualified image paths and labels match the
   validation manifest exactly, with no missing, extra, or repeated paths.
2. Evaluate every unique `probability_class_1` score as a candidate threshold,
   assigning class 1 when `probability_class_1 >= threshold`.
3. Maximize positive-class F1. If candidates have exactly equal maximum F1,
   choose the highest threshold. Freeze this choice before final evaluation.
4. Save the entire sweep, selection criteria, sample count, majority baseline,
   precision, recall/sensitivity, specificity and F1 at both 0.5 and the selected
   threshold, plus SHA-256 hashes of the validation manifest and prediction CSVs.

| Model | Selected threshold | Validation F1 at 0.5 | Validation F1 at selected threshold |
|---|---:|---:|---:|
| Custom CNN | `0.4541136622428894` | 0.177778 | 0.470588 |
| ResNet18 | `0.3251684010028839` | 0.324324 | 0.422535 |

Both selections are unique maxima of their recorded 129-candidate sweeps.
The JSON preserves the original full-precision metrics, prediction hashes and
Colab paths without modification. Validation F1 was used for threshold
selection, so these figures are selection results, not an independent estimate
of generalization. The majority-class validation accuracy baseline is
`104 / 129 = 80.6%`; this is not a model result.

To reproduce the procedure with the original selected checkpoints, generate
validation predictions only and save a separate comparison record:

```bash
python -m src.evaluation --checkpoint checkpoints/custom_cnn/best_model.pth --data-dir data/raw --manifest data/splits/validation.csv --output-dir artifacts/validation/custom_cnn --threshold 0.5
python -m src.evaluation --checkpoint checkpoints/resnet18/best_model.pth --data-dir data/raw --manifest data/splits/validation.csv --output-dir artifacts/validation/resnet18 --threshold 0.5
python -m src.threshold_selection --custom-cnn-predictions artifacts/validation/custom_cnn/predictions.csv --resnet18-predictions artifacts/validation/resnet18/predictions.csv --validation-manifest data/splits/validation.csv --output artifacts/reproduced_validation_thresholds.json
```

The selector refuses to overwrite an existing output file. Keep the checked-in
Colab record frozen; do not replace it with a regenerated or reduced record.
The commands above are reproduction instructions, not a request to retrain or
reselect thresholds. No test predictions enter threshold selection.

### Verify the experimental record without model execution

Check JSON structure, supported criteria, finite thresholds, ordered sweep
candidates, maximum-F1 selection and selected metrics, then verify the local
validation manifest's hash, sample count and majority baseline:

```bash
python -m src.threshold_selection \
  --verify-config reports/validation_thresholds.json \
  --validation-manifest data/splits/validation.csv
```

This is read-only and opens no images or other partition manifests. The local
validation manifest matches the recorded SHA-256. Both source CSVs currently
point to unavailable `/content/drive/MyDrive/...` Colab paths, so the command
reports `source_predictions_verified: false` for each model. The recorded
sweeps and their 0.5 metrics are internally consistent with the validation
class counts, but this does not verify the original prediction bytes or their
association with individual images.

Once the original Colab CSVs have been copied to local ignored artifact paths,
verify their exact recorded hashes, paths/labels, complete recomputed sweeps
and metrics without running either model:

```bash
python -m src.threshold_selection \
  --verify-config reports/validation_thresholds.json \
  --validation-manifest data/splits/validation.csv \
  --custom-cnn-predictions artifacts/validation/custom_cnn/predictions.csv \
  --resnet18-predictions artifacts/validation/resnet18/predictions.csv
```

Each source is verified separately; supplying one CSV leaves the other marked
unverified. Hash checks use file bytes, so reserializing CSVs or changing line
endings will fail verification. Preserve the original files. The loader never
follows recorded Colab artifact paths; explicit local paths are used only by
the verifier.

### Reproducibility and integration limits

The record contains prediction and validation-manifest hashes, but no
checkpoint hashes, selected epochs or Colab runtime/package versions. Local
checkpoints are unavailable, so their pairing with these thresholds and their
compatibility with the repository's checkpoint metadata cannot be confirmed.
Before final evaluation, verify that the original selected weights correspond
to each recorded prediction file; architecture name alone cannot prove this.
The checkpoint loader expects `model_type`, `architecture`, `class_to_idx`
and `model_state_dict` metadata from `src.training`. Evaluation and inference
reject incompatible class mappings and require class 1 to be inhomogeneous.

Dependencies in `requirements.txt` are unpinned, so exact probability
reproduction across runtimes is not guaranteed. Keep the original CSVs,
checkpoint files and environment details outside Git for the audit. Patient
and centre identifiers remain unavailable: validation is provisional at image
level and does not establish patient or centre independence.

Only the aggregate JSON record is tracked for this experiment. Raw data,
checkpoints, `artifacts/`, Colab-style `results/`, and `.env` remain ignored;
do not commit medical images, prediction CSVs or credentials.

## Single-image inference

Use the same frozen configuration for an explicitly supplied image:

```bash
python -m src.inference \
  --checkpoint checkpoints/resnet18/best_model.pth \
  --image /path/to/image.png \
  --threshold-config reports/validation_thresholds.json
```

`CheckpointPredictor` loads the model and threshold once and accepts PIL images
through `predict(image)`. It uses deterministic preprocessing, evaluation mode,
disabled gradients and the same inclusive decision rule as evaluation. The
JSON response includes the probability, predicted class, applied threshold and
configuration path. It does not discover dataset files or read split manifests.

Both pipelines locate the default JSON relative to the source tree, independent
of the working directory. A missing or invalid configuration, unsupported model
name, or selection inconsistent with the recorded sweep causes an error rather
than a fallback to 0.5. Include `reports/validation_thresholds.json` when
packaging the source for inference, or supply its deployed location explicitly.

## Tests

Run all automated tests from the repository root:

```bash
python -m pytest -q
```

The tests cover dataset loading, preprocessing, architectures without weight
downloads, split reproducibility/nonoverlap, label mapping, exact duplicate
grouping, class weights, synthetic-tensor training, checkpoint roundtrip, and
metrics/plots. The held-out medical-image test manifest is not evaluated by
the test suite. Threshold tests cover the unchanged Colab record, malformed
configuration, exact F1 ties, source-hash and metric verification, overwrite
protection, both model lookups, inclusive threshold decisions and evaluation /
inference consistency. All image inputs in pipeline tests are synthetic, and
the inference checkpoint test uses untrained weights without optimization.
To omit even the existing synthetic-training test, run:

```bash
python -m pytest -q -k 'not synthetic_training_and_checkpoint_roundtrip'
```

## Repository layout

```text
src/
  data/       dataset loader, transforms, split and similarity audit
  models/     custom CNN and ResNet18 definitions
  training.py training CLI and checkpoint loader
  evaluation.py checkpoint evaluation CLI and metrics
  inference.py  single-image checkpoint inference
  threshold_config.py    validated shared threshold loader
  threshold_selection.py validation-only selection and record verification
data/splits/  tracked CSV manifests with relative paths and labels
reports/      tracked split/similarity audit and frozen validation threshold record
tests/        automated tests; synthetic training/evaluation only
```

See [AI_USE.md](AI_USE.md) for disclosure of AI-assisted implementation and
the required human review.
