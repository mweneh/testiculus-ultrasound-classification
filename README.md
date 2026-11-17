# TesticulUS Ultrasound Classification

Reproducible binary image-classification pipeline for testicular ultrasound
images. The repository provides a custom CNN and an ImageNet-initialized
ResNet18, fixed image-level data splits, checkpointed training, and
checkpoint-based evaluation.

**No model has been trained as part of the current implementation.** The
reported split and similarity audit are data-preparation results only; there
are no model performance results to report yet.

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

Evaluation is a separate command and requires an explicit manifest. The
predeclared probability threshold is 0.5. If a different threshold is needed,
select it using validation predictions only, record that decision, and pass the
chosen value with `--threshold`; never tune a threshold or select a model on
test results. Run the test command only after model and threshold selection are
complete:

```bash
python -m src.evaluation \
  --checkpoint checkpoints/custom_cnn/best_model.pth \
  --data-dir data/raw \
  --manifest data/splits/test.csv \
  --output-dir artifacts/evaluation/custom_cnn_test \
  --threshold 0.5
```

Evaluation writes `metrics.json`, a per-image `predictions.csv`, and ROC and
precision-recall plots. The positive class is inhomogeneous (1). Report ROC-AUC,
average precision (the precision-recall summary), accuracy, balanced accuracy,
precision, recall/sensitivity, specificity, F1, and the confusion matrix.
Metrics whose required labels or predictions are absent are represented as
`null` and named in `undefined_metrics`.

### Select thresholds from validation predictions

Generate prediction CSVs for both selected checkpoints using the validation
manifest, then choose the threshold maximizing positive-class F1:

```bash
python -m src.evaluation --checkpoint checkpoints/custom_cnn/best_model.pth --data-dir data/raw --manifest data/splits/validation.csv --output-dir artifacts/validation/custom_cnn --threshold 0.5
python -m src.evaluation --checkpoint checkpoints/resnet18/best_model.pth --data-dir data/raw --manifest data/splits/validation.csv --output-dir artifacts/validation/resnet18 --threshold 0.5
python -m src.threshold_selection --custom-cnn-predictions artifacts/validation/custom_cnn/predictions.csv --resnet18-predictions artifacts/validation/resnet18/predictions.csv --validation-manifest data/splits/validation.csv --output reports/validation_thresholds.json
```

The selector verifies that each prediction file has exactly the image paths and
labels in the validation manifest. It evaluates every unique predicted
positive-class probability, maximizes positive-class F1, and breaks ties by
choosing the highest threshold. The JSON records the criteria, threshold sweep,
precision, recall, specificity and F1 at 0.5 and at the selected threshold, and
hashes of both input files. Do not use test predictions to select a threshold.

The majority-class accuracy baseline is calculated from the labels in the
evaluated manifest. For the current validation partition it is `104 / 129 =
80.6%`; this baseline is not a model result. Report class-sensitive metrics
because of the imbalance.

## Tests

Run all automated tests from the repository root:

```bash
python -m pytest -q
```

The tests cover dataset loading, preprocessing, architectures without weight
downloads, split reproducibility/nonoverlap, label mapping, exact duplicate
grouping, class weights, synthetic-tensor training, checkpoint roundtrip, and
metrics/plots. The held-out medical-image test manifest is not evaluated by
the test suite.

## Repository layout

```text
src/
  data/       dataset loader, transforms, split and similarity audit
  models/     custom CNN and ResNet18 definitions
  training.py training CLI and checkpoint loader
  evaluation.py checkpoint evaluation CLI and metrics
data/splits/  tracked CSV manifests with relative paths and labels
reports/      tracked split/similarity audit
tests/        automated tests; synthetic training/evaluation only
```

See [AI_USE.md](AI_USE.md) for disclosure of AI-assisted implementation and
the required human review.
