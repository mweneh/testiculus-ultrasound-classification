# MobileNetV3-Small: post-hoc exploratory extension

This architecture was added after the original Custom CNN and ResNet18 test
outcomes were known. It is an exploratory accuracy–efficiency comparison,
not a prospectively specified third arm or an independent confirmation.
The original models, validation thresholds and results are unchanged.

The completed experiment is recorded in:

- [Protocol declared before training](../reports/mobilenet_exploratory_protocol.json).
- [CPU preflight](../reports/mobilenet_cpu_preflight.json).
- [Complete validation threshold sweep](../reports/mobilenet_v3_small_thresholds.json).
- [Checkpoint and configuration frozen before test](../reports/mobilenet_v3_small_freeze.json).
- [Full comparison and interpretation](../reports/mobilenet_exploratory_summary.md).
- [Full-precision aggregate results and source hashes](../reports/mobilenet_exploratory_results.json).
- [All CPU latency measurements](../reports/model_efficiency.json).

## Architecture, initialization and preprocessing

The implementation uses
[Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
with `MobileNet_V3_Small_Weights.IMAGENET1K_V1`. The final classifier output is
replaced with a two-class linear layer. The official initialization file is
[mobilenet_v3_small-047dcff4.pth](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth);
its downloaded SHA-256 is recorded in the aggregate results.

Labels remain homogeneous = 0 and inhomogeneous = 1. The dataset loader converts
to grayscale; validation/inference resize directly to 224×224 with bilinear
interpolation, repeat grayscale to three channels, convert to [0,1] tensors,
then normalize with ImageNet mean [0.485,0.456,0.406] and standard deviation
[0.229,0.224,0.225]. This preserves the project's ResNet ultrasound convention;
it does not use Torchvision's default ImageNet resize-and-crop recipe.
Only training receives horizontal flips and ±10-degree rotations.

## Completed CPU training

The feature extractor's parameters and BatchNorm statistics were frozen.
Both classifier linear layers were trainable. AdamW used learning rate 0.0001
and weight decay 0.0001, batch size 16, at most 20 epochs, patience 5, seed 42,
two CPU threads and zero loader workers. Weighted cross-entropy used weights
[0.6168032288551331, 2.640350818634033] computed only from the 602 training labels.
The existing validation manifest contains 129 images. Neither split was changed.
The existing validation-loss scheduler was retained.

The actual device was CPU (Intel Core i7-7500U). Training stopped after 10
epochs in 93.44 seconds; epoch durations ranged from 8.83 to 10.19 seconds.
Epoch 5 had minimum validation loss 0.541681. There are 1,519,906 parameters,
of which 592,898 were trainable. Training loss decreased from 0.6767 to 0.5413;
validation loss reached its minimum and then failed to improve for five epochs.
This did not establish clear underfitting requiring another experiment, so
no feature blocks were unfrozen and no additional fine-tuning was performed.

## Commands and completed evaluation order

Use the development environment from the repository root. The commands below
document the completed run; do not repeat the scientific experiment now that
its test result has been observed. Training refuses an existing checkpoint or
history, and threshold/freeze/benchmark creation refuses existing records.

CPU preflight uses random tensors and a disposable copy for the head-training
probe. It never optimizes the existing classifiers or reads dataset images:

```bash
TORCH_HOME=artifacts/torch_cache OMP_NUM_THREADS=2 python -m src.cpu_benchmark \
  --pretrained --probe-training --threads 2 --warmup 10 --repeats 50 \
  --output reports/mobilenet_cpu_preflight.json
```

Training reads only the existing training and validation manifests:

```bash
TORCH_HOME=artifacts/torch_cache OMP_NUM_THREADS=2 python -m src.training \
  --model mobilenet_v3_small --pretrained --freeze-backbone \
  --device cpu --threads 2 --batch-size 16 --learning-rate 0.0001 \
  --epochs 20 --patience 5 --seed 42 --num-workers 0 \
  --data-dir data/raw --split-dir data/splits \
  --output-dir training_runs/mobilenet_v3_small
```

The training summary records timing, parameter counts, device, manifest hashes
and configuration. Raw history and the checkpoint remain in ignored
`training_runs/mobilenet_v3_small/`. Do not commit the checkpoint.

Generate validation probabilities from the best checkpoint before selecting
the threshold. Batch size 1 matches deployed single-image inference and avoids
batch-dependent roundoff at a selected candidate probability:

```bash
OMP_NUM_THREADS=2 python -m src.evaluation \
  --checkpoint training_runs/mobilenet_v3_small/best_model.pth \
  --data-dir data/raw --manifest data/splits/validation.csv \
  --output-dir results/mobilenet_v3_small_validation \
  --threshold 0.5 --batch-size 1 --num-workers 0 --device cpu

OMP_NUM_THREADS=2 python -m src.threshold_selection \
  --mobilenet-predictions results/mobilenet_v3_small_validation/predictions.csv \
  --validation-manifest data/splits/validation.csv \
  --output reports/mobilenet_v3_small_thresholds.json

OMP_NUM_THREADS=2 python -m src.exploratory_evaluation freeze
```

All 129 unique validation probabilities were candidates. The criterion is
maximum positive-class F1 with `p >= threshold`, choosing the highest threshold
on exact F1 ties. The frozen threshold is **0.5038942694664001**; validation
ROC-AUC is 0.6846, AP 0.4631 and selected-threshold F1 0.4878.
The separate version-2 record explicitly discloses the post-hoc status;
`reports/validation_thresholds.json` remains byte-for-byte unchanged.

Only after freezing checkpoint/configuration hashes was this command run once:

```bash
OMP_NUM_THREADS=2 python -m src.exploratory_evaluation evaluate-once
```

The wrapper reserves `results/mobilenet_v3_small_test/` before reading test
images and stores an evaluation receipt. Reusing that directory is refused,
even after a partial failure. Do not bypass it with a different output directory
or rerun the generic evaluation CLI on test data. The single test result is
ROC-AUC 0.6562, AP 0.4889, accuracy 0.8217, balanced accuracy 0.6463,
sensitivity 0.3600, specificity 0.9327 and F1 0.4390, with confusion matrix
`[[97,7],[16,9]]`. No tuning followed this evaluation.

The final efficiency comparison uses the trained checkpoints on the same CPU,
float32, batch size 1, 224×224 inputs, two intra-op threads, one inter-op thread,
10 warm-up forwards and 50 measurements per model. Model-forward latency
excludes decoding, preprocessing, file loading and API overhead:

```bash
OMP_NUM_THREADS=2 python -m src.cpu_benchmark \
  --mobilenet-checkpoint training_runs/mobilenet_v3_small/best_model.pth \
  --threads 2 --warmup 10 --repeats 50 --output reports/model_efficiency.json
python -m src.model_comparison
```

Rendering the existing summary is safe without images, checkpoints or raw
prediction CSVs, and does not repeat evaluation:

```bash
python -m src.model_comparison \
  --from-report reports/mobilenet_exploratory_results.json \
  --output artifacts/mobilenet_comparison.md
```

## Inference, API, Grad-CAM and Docker

Inference restores the complete trained checkpoint without downloading ImageNet
weights. MobileNet automatically uses `reports/mobilenet_v3_small_thresholds.json`;
the two original models retain their original threshold file and thresholds.

```bash
python -m src.inference \
  --checkpoint training_runs/mobilenet_v3_small/best_model.pth \
  --image /path/to/image.png --device cpu

python -m src.gradcam \
  --checkpoint training_runs/mobilenet_v3_small/best_model.pth \
  --image /path/to/image.png --output-dir artifacts/gradcam/mobilenet_v3_small
```

Grad-CAM targets the final spatial feature block, `features.12`, and supports
the frozen feature extractor. The default explained class follows the frozen
decision threshold. Dataset-derived visualizations remain adaptations under
the [dataset attribution and ShareAlike terms](../data/README.md).

The original ResNet18 demo commands still work. To additionally enable MobileNet:

```bash
export MOBILENET_V3_SMALL_CHECKPOINT="$PWD/training_runs/mobilenet_v3_small/best_model.pth"
unset THRESHOLD_CONFIG
export DEVICE=cpu
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Keep `RESNET18_CHECKPOINT` and/or `CUSTOM_CNN_CHECKPOINT` set to serve them too.
Unsetting a global `THRESHOLD_CONFIG` allows automatic per-model lookup; an
explicit global override must contain the requested model. Predictions use
`POST /predict?model_type=mobilenet_v3_small` with multipart field `file`.
There is no change to the existing requested/default demonstration model.

The Docker image includes both threshold records but no checkpoints. Use a
separate tag and explicitly opt into the third checkpoint:

```bash
docker build -t testiculus-inference:mobilenet .
docker run --rm --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$PWD/training_runs,dst=/models,readonly" \
  -e RESNET18_CHECKPOINT=/models/resnet18/best_model.pth \
  -e MOBILENET_V3_SMALL_CHECKPOINT=/models/mobilenet_v3_small/best_model.pth \
  testiculus-inference:mobilenet
```

The synthetic container check can include all three trained models:

```bash
OMP_NUM_THREADS=1 python -m src.container_smoke \
  --image-tag testiculus-inference:mobilenet --include-mobilenet \
  --checkpoints-dir training_runs --output artifacts/mobilenet_container_verification.json
```

MobileNet's checkpoint is locally trained in this exploratory run, rather than
one of the original Google Drive artifacts. Deployment requires supplying it
separately and setting the optional environment variable above. It is not
automatically substituted for ResNet18.

## Interpretation and limitations

MobileNet is faster and smaller than ResNet18 in this benchmark, but has more
parameters than the Custom CNN. The test sensitivity is low, and its ROC-AUC/AP
are below ResNet18's. These results do not establish overall superiority or
statistical significance. Timings are specific to this hardware and runtime;
they are not API latency measurements. No GPU latency or energy use was measured.
Only one seed and one frozen-backbone configuration were run. The reused test
partition, absent verified patient/centre mappings and unresolved local release
version limit scientific conclusions. There is no clinical-readiness claim.

## Verification

The complete suite passed **150 tests** with no deselections and one Starlette
TestClient deprecation warning:

```bash
OMP_NUM_THREADS=1 python -m pytest -q
```

This includes construction/output shape, both initialization options, exact
preprocessing, frozen feature parameters and BatchNorm statistics, trainable
head updates, checkpoint restoration, timing, threshold selection, API,
Grad-CAM, one-evaluation guards and preservation of the original metrics.
Unit tests use synthetic images and temporary checkpoints. Pretrained-download
requests are intercepted in unit tests; the actual ImageNet initialization was
downloaded for the preflight and training run. The legacy synthetic Custom CNN
training test is included now; it does not retrain the original research model.

With the actual trained checkpoint, API probabilities matched local inference
exactly on the validation sample defining the threshold. Reproduction of its
saved probability differed by 1.19e-7 with one CPU thread instead of two; the
frozen-threshold decision agreed. Grad-CAM on synthetic pixels produced finite,
nonconstant 224×224 maps and preserved predictions/parameter gradients. See
[the application check](../reports/mobilenet_application_verification.json).

The separate Docker image passed the all-three-model synthetic API and Grad-CAM
check, with read-only checkpoints, a read-only root filesystem, non-root user
and successful health checks. The temporary container was removed. Evidence is
in [the container report](../reports/mobilenet_container_verification.json).
The successful build reused the previously verified CPU dependency cache;
package versions did not change. An uncached attempt timed out downloading the
large PyTorch CPU wheel, so a fresh machine still needs reliable network access
for the image/dependency downloads. The original `testiculus-inference:local`
tag was preserved; the verified image also has the documented `mobilenet` tag.
