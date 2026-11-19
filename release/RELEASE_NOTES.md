# Frozen TesticulUS research checkpoints — models-v1.0.0

Prepared locally, not published. Intended use: noncommercial research and
academic reproducibility. **Checkpoint redistribution/licensing is pending
owner review**; see `WEIGHT_NOTICES.md`. These artifacts are not clinically
validated or suitable for diagnosis. No new training, threshold optimization
or held-out evaluation was performed to prepare this release.

## Models and frozen decisions

| Checkpoint asset | Model | Positive-class threshold | Status |
|---|---|---:|---|
| `testiculus-custom-cnn-v1.pth` | Custom CNN, grayscale, trained from scratch | 0.4541136622428894 | Original frozen Colab experiment |
| `testiculus-resnet18-v1.pth` | ResNet18, ImageNet initialization, full fine-tuning | 0.3251684010028839 | Original frozen Colab experiment |
| `testiculus-mobilenet-v3-small-v1.pth` | MobileNetV3-Small, ImageNet initialization, frozen features/BatchNorm | 0.5038942694664001 | Post-hoc exploratory extension |

The files are byte-for-byte copies of the final trained checkpoints. They retain
complete trained states and checkpoint metadata; they are not new exports,
quantized models, TorchScript files or initialization-only weights. Loading uses
the source repository's `src.training` / `src.inference` interfaces with
`weights_only=True`, strict state loading and no ImageNet download.

`validation_thresholds.json` contains the complete original two-model record;
`mobilenet_v3_small_thresholds.json` contains the separate exploratory record.
Both retain prediction hashes, full validation sweeps, selection criteria and
metrics. Thresholds maximize positive-class F1 over unique validation scores,
with the highest threshold winning exact ties; prediction uses `p >= threshold`.
No test-based threshold selection is introduced.

## Saved evaluation evidence and limits

All models used the same 129 image-level test samples (104 homogeneous, 25
inhomogeneous). These are saved historical results, not a new evaluation:

| Model | ROC-AUC | AP | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 0.6123 | 0.4029 | 0.6124 | 0.5317 | 0.4000 | 0.6635 | 0.2857 |
| ResNet18 | 0.7473 | 0.5220 | 0.6434 | 0.6421 | 0.6400 | 0.6442 | 0.4103 |
| MobileNetV3-Small | 0.6562 | 0.4889 | 0.8217 | 0.6463 | 0.3600 | 0.9327 | 0.4390 |

Majority-class accuracy is 80.62%, with zero positive detections. MobileNet was
added after the original test outcomes were known; its own validation threshold
was frozen before its test evaluation. The expanded comparison is exploratory.
There is no claim of statistical superiority, probability calibration,
patient-independent evaluation, centre-held-out evaluation or clinical validity.
Only one seed and 25 positive validation/test images limit inference. Verified
patient/examination and per-image centre mappings were unavailable, leaving
possible within-examination correlation across image-level partitions.

The local labelled data contain 860 grayscale 300×300 images: 696 homogeneous
(label 0) and 164 inhomogeneous (positive label 1), split 602/129/129. Another
200 unlabelled images were excluded from supervised classification. The local
archive count (1,060) differs from the provider's 1,053 and is not version-matched.
The original validation metric JSONs' dataset-wide majority baseline differs
from the partition baseline; model metrics and frozen threshold selection are
unaffected. Original records remain unchanged.

## Preprocessing and installation

All models convert to grayscale and resize directly to 224×224 with bilinear
interpolation. Custom CNN uses one channel scaled to [0,1]. ResNet18/MobileNet
repeat grayscale to three channels and normalize by ImageNet mean
[0.485,0.456,0.406] / standard deviation [0.229,0.224,0.225]. Evaluation/inference
have no stochastic augmentation. Softmax class 1 is an image-class score,
not a validated diagnostic risk.

Use the matching source checkout, its README and `src.release_assets` commands.
The manifest pins repository `mweneh/testiculus-ultrasound-classification`, tag
`models-v1.0.0`, exact asset names, byte sizes and SHA-256 hashes. The optional
public-release downloader verifies against this trusted source manifest,
including notices, and installs to a new ignored directory. It does not replace
existing files. Until an authorized release exists, GitHub download URLs are
prospective and may return 404. No live release download is claimed verified.

`SHA256SUMS` covers every payload file, including `release_manifest.json`, but
not itself. Verify locally with `sha256sum -c SHA256SUMS`. Hashes support byte
integrity, not independent provenance authentication or rights clearance.
Use the pinned source manifest as the trust reference; a checksum downloaded
from the same untrusted location is insufficient by itself.

## Dataset attribution and redistribution notices

Credit: Morelli et al. and the TesticulUS dataset contributors.
Dataset: https://ditto.ing.unimore.it/testiculus/.
Licence: CC BY-NC-SA 4.0,
https://creativecommons.org/licenses/by-nc-sa/4.0/.

Cite both publications. `DATASET_REFERENCES.bib` preserves the supplied entries;
the ICIAP citation below uses the final publisher metadata rather than that
entry's provisional year and pagination:

1. Morelli, N., Marchesini, K., Lumetti, L., Santi, D., Grana, C., and Bolelli, F.
   (2026). *Enhancing Testicular Ultrasound Image Classification Through Synthetic
   Data and Pretraining Strategies*. Image Analysis and Processing – ICIAP 2025,
   Springer, pp. 26–39. https://doi.org/10.1007/978-3-032-10192-1_3.
   Version of record published 2 January 2026 for the ICIAP 2025 conference:
   https://link.springer.com/chapter/10.1007/978-3-032-10192-1_3.
2. Morelli, N., Marchesini, K., Santi, D., Grana, C., and Bolelli, F. (2026).
   *A New Multicenter Testicular US Dataset and a Lightweight Cond-UNet for
   Generalization in US Segmentation*. Proceedings of the British Machine Vision
   Conference. Cond-UNet is a segmentation architecture, not this classifier.

Original ultrasound images and adaptations, including Grad-CAM overlays, require
attribution, change notices and applicable noncommercial/ShareAlike compliance.
No original images, contact sheets, predictions CSVs, credentials or optimizer
state exports are added by this release packaging process. The original
checkpoints are preserved without reserialization. Dataset licensing is distinct
from independently developed software licensing and pretrained-weight terms;
see `WEIGHT_NOTICES.md` and the included Torchvision software notice for the
unresolved checkpoint-redistribution review. No blanket model-weight licence is
asserted by these notes.
