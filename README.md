# TesticulUS Ultrasound Classification

Reproducible binary image-classification pipeline for testicular ultrasound
images. The repository provides a custom CNN and an ImageNet-initialized
ResNet18, fixed image-level data splits, checkpointed training, and
checkpoint-based evaluation and single-image inference.

The original Custom CNN and ResNet18 completed training, validation-based
threshold selection and final held-out evaluation in Google Colab. **Those
experiments remain frozen.** Their checkpoints, thresholds and results are
unchanged. A separately recorded MobileNetV3-Small experiment was subsequently
authorized and completed as a **post-hoc exploratory extension**.

The [experiment summary](reports/experiment_summary.md) separates validation
selection from final test results. The complete validation record is preserved
unchanged in [validation_thresholds.json](reports/validation_thresholds.json);
[final_test_results.json](reports/final_test_results.json) preserves the supplied
test summaries and their provenance at import. Original trained weights and
detailed artifacts are now available locally under ignored `training_runs/`
and `results/`, copied from `MyDrive/Capstone/`. The aggregate
[verification report](reports/artifact_verification.json) records source hashes,
full-precision metrics, checkpoint metadata and successful trained-model checks.
Weights, prediction rows and medical images remain outside Git.

This is a research application. Patient and acquisition-centre independence
are unverified; the system is not clinically validated or suitable for diagnosis.

## Frozen final test results

Both models used the same 129 images: 104 homogeneous and 25 inhomogeneous.
Class 1 is positive; confusion matrices use actual rows and predicted columns,
ordered [0, 1]. Thresholds were selected on validation only.

| Model | ROC-AUC | Average precision | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 0.6123 | 0.4029 | 0.6124 | 0.5317 | 0.4000 | 0.6635 | 0.2857 |
| ResNet18 | 0.7473 | 0.5220 | 0.6434 | 0.6421 | 0.6400 | 0.6442 | 0.4103 |

- Custom CNN: `[[69,35],[15,10]]`, threshold `0.4541136622428894`.
- ResNet18: `[[67,37],[9,16]]`, threshold `0.3251684010028839`.

Both accuracies fall below the **80.62% majority-class baseline**. ResNet18 has
higher reported class-sensitive and ranking metrics; statistical significance
has not been established. An always-homogeneous classifier has zero sensitivity
and balanced accuracy 0.5.

These are user-supplied Colab results, rounded to four decimal places.
All supplied values agree with the original test metric JSONs at the stated
precision. All full-precision test metrics, including ROC-AUC/AP, also reproduce
from the original saved prediction CSVs at the frozen thresholds. The two CSVs
contain identical test paths and labels. This verification did not open test
images or the test manifest, or rerun held-out model inference.

One original-artifact discrepancy is preserved and flagged: both validation
`metrics.json` files record majority baseline `696/860 = 80.93%` instead of
`104/129 = 80.62%`. Their model metrics reproduce. The frozen threshold JSON and
final test artifacts already have the correct partition baseline, and the
F1-based selections are unaffected. No original artifact was edited.

Regenerate the summary without model execution or opening dataset partitions:

```bash
python -m src.experiment_summary --output reports/experiment_summary.md
```

Compare the original full-precision metrics with the supplied summaries and
print artifact SHA-256 hashes:

```bash
python -m src.experiment_summary \
  --custom-cnn-metrics results/custom_cnn_test/metrics.json \
  --resnet18-metrics results/resnet18_test/metrics.json \
  --output artifacts/verified_experiment_summary.md
```

This verifier expects the `src.evaluation` JSON structure, checks sample counts,
thresholds, confusion matrices and all supplied metrics at the stated precision,
and never changes the frozen records. The separate complete verification command
below also recomputes metrics from prediction CSVs without final held-out model
execution.

## Exploratory MobileNetV3-Small extension

MobileNet was added after the original test outcomes were known, so it is not a
prospectively specified third comparison arm. Its ImageNet-initialized feature
extractor and BatchNorm statistics were frozen while the classification head
was trained locally on CPU. Early stopping completed 10 epochs in 93.44 seconds
and selected epoch 5. The validation-only frozen threshold is
`0.5038942694664001`, stored separately in
[mobilenet_v3_small_thresholds.json](reports/mobilenet_v3_small_thresholds.json).

After configuration freeze, one test evaluation produced ROC-AUC **0.6562**,
AP **0.4889**, accuracy **0.8217**, balanced accuracy **0.6463**, sensitivity
**0.3600**, specificity **0.9327** and F1 **0.4390**; confusion matrix
`[[97,7],[16,9]]`. No tuning followed. Original model results above were preserved.

The [comparison report](reports/mobilenet_exploratory_summary.md) includes model
size, parameters and CPU latency measured under common conditions for all three
trained checkpoints. The [experiment guide](docs/mobilenet_experiment.md)
documents the CPU configuration, benchmark, training/evaluation order, source
hashes and optional API/Docker deployment. MobileNet does not replace the
existing ResNet18 demonstration. Its sensitivity is low; the exploratory result
does not establish overall superiority or clinical suitability.

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

### Dataset attribution

TesticulUS is credited to Morelli et al. and the dataset contributors.

- Dataset: [official TesticulUS page](https://ditto.ing.unimore.it/testiculus/).
- Licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Cite both required publications: [Morelli et al., ICIAP 2025](https://federicobolelli.it/pub_files/2025iciap.pdf)
  and [Morelli et al., BMVC 2026](https://iris.unimore.it/handle/11380/1416328).
  The provider's supplied entries are preserved in [references.bib](data/references.bib)
  with keys `2025ICIAP` and `2026BMVC`.

The licence permits noncommercial sharing and adaptation with attribution and
ShareAlike. Original dataset images and adapted ultrasound visualizations are
subject to CC BY-NC-SA 4.0. Dataset-based Grad-CAM visualizations, overlays and
other modified images are adaptations: when sharing them, credit the creators,
link the dataset and licence, identify changes, cite both papers and apply the
applicable ShareAlike terms. Independently developed software is governed by
its own applicable licence; this dataset notice does not relicense the code.
See the [dataset documentation](data/README.md) for attribution examples and
the separate software-licence status.

### Provenance and limitations

The local directory structure is labelled TesticulUS, but this checkout has no
archive checksum, release identifier or provider receipt establishing the exact
source version. The local copy contains 1,060
images across `homogeneous/`, `inhomogeneous/`, and `imgs/` (860 labelled and
200 unlabelled). The [official dataset page](https://ditto.ing.unimore.it/testiculus/)
now describes TesticulUS-Real as 1,053 real images and also describes the
separate synthetic TesticulUS-Syn release. The local copy therefore is not
treated as verified byte-for-byte identical to that public release. This
unresolved version match is separate from the official dataset's documented
CC BY-NC-SA 4.0 licence.

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

Historical split-generation command, from the repository root. Do not rerun
it on the frozen experiment partitions:

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

Historical contact-sheet command, which reads dataset images. It is excluded
from the frozen application checks:

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

The application was verified on Linux with Python 3.12.3. The pinned serving
dependencies require Python 3.12 or later. Use a fresh virtual environment
without `--system-site-packages`; reusing an environment with unrelated tools
can introduce dependency conflicts. For inference and the API, use the smaller
[serving installation](#fastapi-application). For development and tests:

```bash
python3.12 -m venv .venv
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
model checkpoints. Keeping dataset images out of Git is this repository's
source-control policy, not a prohibition on sharing under the dataset licence.

The exploration notebook is tracked with saved outputs and execution counts
cleared. Before committing an executed notebook, clear all outputs and remove
attachments: ultrasound displays and image grids can be embedded inside the
notebook even when no image files are staged. The repository hygiene test checks
this. Earlier Git commits still contain the old embedded displays; clearing the
current notebook does not remove those historical objects. History has not been
rewritten.

## Training implementation (historical reproduction only)

The original Colab training runs are complete. The commands in this section
document the implementation; do not rerun training for application development.

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
  --checkpoint training_runs/custom_cnn/best_model.pth \
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
python -m src.evaluation --checkpoint training_runs/custom_cnn/best_model.pth --data-dir data/raw --manifest data/splits/validation.csv --output-dir artifacts/validation/custom_cnn --threshold 0.5
python -m src.evaluation --checkpoint training_runs/resnet18/best_model.pth --data-dir data/raw --manifest data/splits/validation.csv --output-dir artifacts/validation/resnet18 --threshold 0.5
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
validation manifest matches the recorded SHA-256. Without explicit prediction
paths this command checks only the record and manifest and reports
`source_predictions_verified: false`; it does not search recorded Colab paths.

Both original validation CSVs are now present under `results/`. Their exact
recorded hashes, paths/labels, complete sweeps and selected-threshold metrics
have been verified. Reproduce this check without running either model:

```bash
python -m src.threshold_selection \
  --verify-config reports/validation_thresholds.json \
  --validation-manifest data/splits/validation.csv \
  --custom-cnn-predictions results/custom_cnn_validation/predictions.csv \
  --resnet18-predictions results/resnet18_validation/predictions.csv
```

Each source is verified separately; supplying one CSV leaves the other marked
unverified. Hash checks use file bytes, so reserializing CSVs or changing line
endings will fail verification. Preserve the original files. The loader never
follows recorded Colab artifact paths; explicit local paths are used only by
the verifier.

### Reproducibility and integration limits

The downloaded checkpoints load with the correct class mapping and preprocessing
metadata. Their recorded best epochs/losses agree with the training histories:
Custom CNN completed 15 epochs and selected epoch 8; ResNet18 completed 14 and
selected epoch 7. For two validation samples per model, local probabilities
agree with Colab within `2.1e-7`, below the predefined absolute tolerance `1e-5`;
frozen-threshold decisions agree. API and CLI predictions agree exactly. Grad-CAM
was verified with both trained models on synthetic inputs without changing
predictions or parameter gradients.

These are compatibility and reproducibility checks, not new test evaluations.
The project developer supplied the trained checkpoints and experiment artifacts
from their Google Drive folder `MyDrive/Capstone/`, mirrored locally as:

```text
training_runs/custom_cnn/best_model.pth
training_runs/custom_cnn/history.csv
training_runs/resnet18/best_model.pth
training_runs/resnet18/history.csv
results/custom_cnn_test/{metrics.json,predictions.csv}
results/resnet18_test/{metrics.json,predictions.csv}
results/custom_cnn_validation/
results/resnet18_validation/
results/validation_thresholds.json
```

### Weight sources

The Custom CNN was initialized from scratch. The repository's ResNet18 training
path uses Torchvision's `ResNet18_Weights.DEFAULT`, which resolves to
`IMAGENET1K_V1` in the verified Torchvision 0.29.1 installation. These are
[official ImageNet-1K pretrained weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
served as [resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth).
The training code replaces the 1,000-class head with a two-class head before
fine-tuning. The Colab checkpoint records `pretrained_initialization: true`,
but does not record the original Torchvision version or pretrained-file hash;
the precise original download cannot be independently authenticated.

The trained ultrasound classifiers are the developer-supplied
`training_runs/custom_cnn/best_model.pth` and
`training_runs/resnet18/best_model.pth` above. They are not distributed in Git
or the Docker image, and there is no automatic checkpoint download. Obtain
them separately from the project developer. Inference restores these complete
trained checkpoints with `pretrained=False`; it does not download ImageNet
weights. Preprocessing follows the recorded ultrasound-training transforms,
including direct 224×224 resizing, rather than the default ImageNet crop recipe.

The third checkpoint, `training_runs/mobilenet_v3_small/best_model.pth`, was
trained locally for the documented exploratory extension, initialized from
Torchvision's `MobileNet_V3_Small_Weights.IMAGENET1K_V1`. Its source, configuration
and weight hashes are preserved in
[the exploratory record](reports/mobilenet_exploratory_results.json); it is also
excluded from Git and must be supplied separately for deployment.

The downloaded threshold file is byte-identical to the unchanged tracked record.
Checkpoint, history, metric and prediction-file hashes are recorded in
`reports/artifact_verification.json`; they fingerprint supplied local copies and
do not substitute for an independent provider checksum. The original rounded
test-summary JSON preserves its provenance status at import; subsequent verified
evidence is kept separately rather than rewriting the experimental record.
The checkpoint loader expects `model_type`, `architecture`, `class_to_idx`
and `model_state_dict` metadata from `src.training`. Evaluation and inference
reject incompatible class mappings and require class 1 to be inhomogeneous.

Research dependencies in `requirements.txt` are unpinned. The separate serving
requirements record the direct versions used for application checks, and the
Docker base image is pinned by digest; these do not reconstruct the missing
Colab environment. Exact probabilities across runtimes are not guaranteed.
Keep the original CSVs, checkpoint files and environment details outside Git
for the audit. Patient
and centre identifiers remain unavailable: validation is provisional at image
level and does not establish patient or centre independence.

Only aggregate experiment records and documentation are tracked. Raw data,
checkpoints, `training_runs/`, `artifacts/`, Colab-style `results/`, and `.env`
remain ignored; do not commit medical images, prediction CSVs or credentials.

## Single-image inference

Use the same frozen configuration for an explicitly supplied image:

```bash
python -m src.inference \
  --checkpoint training_runs/resnet18/best_model.pth \
  --image /path/to/image.png \
  --threshold-config reports/validation_thresholds.json
```

`CheckpointPredictor` loads the model and threshold once and accepts PIL images
through `predict(image)`. It uses deterministic preprocessing, evaluation mode,
disabled gradients and the same inclusive decision rule as evaluation. The
JSON response includes the probability, predicted class, applied threshold and
configuration path. It does not discover dataset files or read split manifests.

Both pipelines locate the model-specific default JSON relative to the source
tree, independent of the working directory. A missing or invalid configuration,
unsupported model name, or selection inconsistent with the recorded sweep causes an error rather
than a fallback to 0.5. Include `reports/validation_thresholds.json` when
packaging the source for inference, or supply its deployed location explicitly.
MobileNet additionally requires `reports/mobilenet_v3_small_thresholds.json`.

## FastAPI application

For a CPU serving environment, install the application versions separately
from the historical research environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-serving.txt
python -m pip check
```

This installation was repeated in an isolated Python 3.12 environment:
dependency checks, both inference/Grad-CAM CLIs and HTTP API predictions passed
with the trained checkpoints and synthetic images. The pre-existing development
environment inherits system packages and has unrelated dependency conflicts;
use a fresh environment or the Docker image for serving. The Docker dependency
check also passed.

After downloading the original checkpoints, start the service:

```bash
export CUSTOM_CNN_CHECKPOINT="$PWD/training_runs/custom_cnn/best_model.pth"
export RESNET18_CHECKPOINT="$PWD/training_runs/resnet18/best_model.pth"
export THRESHOLD_CONFIG="$PWD/reports/validation_thresholds.json"
export DEVICE=cpu
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Configure either or both original models. MobileNet is optional; see the
[extension deployment commands](docs/mobilenet_experiment.md#inference-api-grad-cam-and-docker)
for its checkpoint variable and model-specific threshold lookup.
Each configured checkpoint is loaded once
during application lifespan startup; missing files, incompatible checkpoint
metadata or invalid threshold configuration fail startup. Random weights are
used only by tests and are never a service fallback. `create_app(...)` also
accepts explicit checkpoint paths, configuration and device for integration.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail -F 'file=@/path/to/image.png' \
  'http://127.0.0.1:8000/predict?model_type=resnet18'
```

`POST /predict` requires `model_type=custom_cnn`, `model_type=resnet18` or
`model_type=mobilenet_v3_small` and a
multipart `file`. It returns `model_type`, `prediction` (0 or 1), `class_name`,
`positive_class`, `probability_class_1`, the frozen `threshold` and its
`threshold_source`. There is no request-level threshold override. `/health`
reports loaded models and thresholds; `/docs` provides the OpenAPI interface.
An unconfigured model returns 503; invalid model parameters/missing files return
422; empty/undecodable images return 400; unsupported formats return 415; images
over 5 MiB or 20 megapixels return 413. The application does not persist uploads.

API and CLI share `CheckpointPredictor` and deterministic preprocessing:

| Step | Custom CNN | ResNet18 / MobileNetV3-Small |
|---|---|---|
| Source conversion | Grayscale L | Grayscale L |
| Resize | 224×224, bilinear | 224×224, bilinear |
| Model channels | 1 | Grayscale repeated to 3 channels |
| Tensor values | [0, 1] | [0, 1], then ImageNet normalization |
| Mean / standard deviation | None | [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225] |

Inference uses evaluation mode and disabled gradients. A per-model lock
serializes shared model execution. See the official FastAPI documentation for
[lifespan loading](https://fastapi.tiangolo.com/advanced/events/) and
[multipart uploads](https://fastapi.tiangolo.com/tutorial/request-files/).

## Grad-CAM

Generate explanations for an explicitly supplied image without training:

```bash
python -m src.gradcam \
  --checkpoint training_runs/resnet18/best_model.pth \
  --image /path/to/image.png \
  --output-dir artifacts/gradcam/resnet18
```

The implementation follows [Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391):
differentiate the target class's pre-softmax logit, spatially average its
gradients, weight activation channels, sum and apply ReLU. Target layers are
`features.12` (Custom CNN's last convolution) and `layer4.1` (ResNet18's final
residual block); MobileNet uses its final spatial feature block, `features.12`.
The heatmap is interpolated to model input size and normalized
to [0, 1]; constant maps become zeros. Temporary hooks are removed, model modes
are restored and weights/parameter gradients remain unchanged.

By default, the explained class is the frozen-threshold prediction, which can
differ from the softmax argmax. Use `--target-class 0` or `--target-class 1` to
inspect a specific class. Outputs are `heatmap.png` (224×224 grayscale),
`overlay.png` (red attribution at original image size) and `prediction.json`
with the threshold, prediction, explained class and target layer. Generated
overlays may reproduce source images and belong under ignored `artifacts/`.
Grad-CAM visualizations derived from TesticulUS images are adaptations covered
by the [dataset attribution and ShareAlike requirements](#dataset-attribution).
These are coarse attribution maps, not segmentation masks, verified lesion
localization or evidence of clinical validity.

## Docker

Build the CPU inference image:

```bash
docker build -t testiculus-inference:local .
```

The build context allows only source code, serving requirements, Dockerfile
and the two model-specific frozen threshold records. Datasets, checkpoints,
histories, prediction artifacts, credentials, Git metadata and virtual
environments are excluded.
The image contains no trained weights. Mount downloaded checkpoints read-only:

```bash
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$PWD/training_runs,dst=/models,readonly" \
  -e CUSTOM_CNN_CHECKPOINT=/models/custom_cnn/best_model.pth \
  -e RESNET18_CHECKPOINT=/models/resnet18/best_model.pth \
  testiculus-inference:local
```

The service runs as UID/GID 10001, with one worker and a `/health` Docker
health check. Mounted files must be readable by that UID. The default device
is CPU; this image does not configure CUDA. Use the same health and prediction
requests shown above.

Reproduce the container acceptance checks with the downloaded trained weights:

```bash
python -m src.container_smoke \
  --checkpoints-dir training_runs \
  --output artifacts/container_verification.json
```

This starts a temporary localhost-only container on an ephemeral port, checks
both trained-model predictions against local inference, checks Grad-CAM and
health, verifies read-only mounts/non-root execution and artifact exclusions,
then stops/removes the container. Inputs are synthetic; it opens no medical
images. The host needs the test/development environment and Docker access.
The [container acceptance report](reports/container_verification.json) records
a successful run with both downloaded trained checkpoints: API probabilities
match local CPU inference exactly, both Grad-CAM commands work, and Docker
health/non-root/read-only checks pass. The temporary container was removed.

## Reproduce original-artifact acceptance

From the development environment with `requirements.txt` installed:

```bash
python -m src.artifact_verification \
  --artifact-root . \
  --data-dir data/raw \
  --check-api \
  --output artifacts/artifact_verification.json
```

This reads original validation/test prediction CSVs and metric JSONs, verifies
the threshold record and validation manifest, checks checkpoint metadata and
histories, and recomputes CSV metrics at the already frozen thresholds. It also
checks trained-model inference on one validation image from each class per model
and compares API/CLI outputs. Test images and the test manifest are never opened;
there is no model training, threshold search on test results or new test inference.
Omit `--data-dir` to use synthetic images for the inference checks.

The tracked aggregate acceptance report contains complete test metrics and
artifact fingerprints without image identifiers, prediction rows or weights.
Original validation baseline discrepancies are reported explicitly; all other
metric inconsistencies fail verification. Review a regenerated report before
replacing tracked evidence. Original Colab runtime/package versions and verified
patient/centre metadata remain unavailable.

## Tests

Install `requirements.txt` for the development suite (including pytest and
HTTPX). From the repository root, run the application checks without any
optimization steps:

```bash
python -m pytest -q -k 'not synthetic_training_and_checkpoint_roundtrip'
```

The tests cover dataset loading, preprocessing, architectures without weight
downloads, split reproducibility/nonoverlap, label mapping, exact duplicate
grouping, class weights, checkpoint loading/serialization, and
metrics/plots. The held-out medical-image test manifest is not evaluated by
the test suite. Threshold tests cover the unchanged Colab record, malformed
configuration, exact F1 ties, source-hash and metric verification, overwrite
protection, both model lookups, inclusive threshold decisions and evaluation /
inference consistency. All image inputs in pipeline tests are synthetic, and
the inference checkpoint test uses untrained weights without optimization.
The excluded test is
`tests/test_training.py::test_synthetic_training_and_checkpoint_roundtrip`.
It runs one optimization epoch on a newly initialized Custom CNN using random
16×16 tensors, then checks checkpoint restoration in a temporary directory.
It uses neither ultrasound images nor the trained Colab checkpoints. It was
deselected to keep application verification free of optimization; it is not a
reported failure. An unfiltered `python -m pytest -q` includes that test and
performs the synthetic optimization, so it was not run during the initial
two-model application checks. It is included in the MobileNet extension checks.

API tests use FastAPI TestClient with actual untrained checkpoint
roundtrips for all three architectures; no ImageNet downloads or optimization occur.
They cover upload validation, startup configuration, model-specific decisions,
readiness and API/CLI probability agreement. Grad-CAM tests cover analytic
attribution, all three architectures, constant maps, mode/gradient preservation,
hook cleanup and CLI output alignment. Summary tests check supplied metrics
against confusion counts and frozen thresholds. The serving tests validate
software behavior. The separate artifact acceptance checks reproduce saved
Colab metrics from prediction CSVs and verify trained-model compatibility.

The earlier application run on 18 November 2026 had **128 passed, 1 deselected**.
After adding the notebook hygiene regression test, that phase had
**129 passed, 1 deselected** (130 collected). It emits one Starlette warning
that its HTTPX-based TestClient integration is deprecated; the API tests pass.
These counts do not include the separate original-artifact acceptance or
running-container smoke checks, which also passed with the trained weights.
The only original metric discrepancy is the documented validation majority
baseline; no threshold, weight, prediction CSV or original metric file changed.

The MobileNet extension adds architecture/initialization, frozen-backbone and
BatchNorm preservation, CPU timing, checkpoint, preprocessing, validation
selection, API, Grad-CAM and single-evaluation safeguards. Its complete test run
uses `python -m pytest -q`, including the legacy synthetic-training test and
the new synthetic MobileNet optimization test. These train fresh temporary
models only; the original research checkpoints are not retrained. The current
full suite has **150 passed**, with one Starlette TestClient deprecation warning
and no deselections. The earlier exclusion and 128/129 counts above describe
the initial frozen two-model application phase.

## Repository layout

```text
src/
  data/       dataset loader, transforms, split and similarity audit
  models/     custom CNN and ResNet18 definitions
  training.py training CLI and checkpoint loader
  evaluation.py checkpoint evaluation CLI and metrics
  inference.py  single-image checkpoint inference
  api.py        FastAPI lifespan, health and prediction routes
  gradcam.py    Grad-CAM and overlay CLI
  experiment_summary.py frozen result checks and summary generator
  artifact_verification.py original artifacts and trained-model acceptance
  cpu_benchmark.py CPU forwards and disposable synthetic head-training probe
  exploratory_evaluation.py freeze evidence and guard one exploratory test run
  model_comparison.py saved exploratory metrics and efficiency comparison
  container_smoke.py Docker acceptance with synthetic pixels / trained weights
  threshold_config.py    validated shared threshold loader
  threshold_selection.py validation-only selection and record verification
data/splits/  tracked CSV manifests with relative paths and labels
reports/      tracked split/similarity audit and frozen validation threshold record
Dockerfile    CPU serving image; checkpoints mounted separately
tests/        automated tests; synthetic training/evaluation only
```

See [AI_USE.md](AI_USE.md) for disclosure of AI-assisted implementation and
the required human review.
