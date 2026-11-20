---
title: TesticulUS Research Prototype
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
startup_duration_timeout: 10m
---

# TesticulUS research prototype

**Research and educational use only. Not for clinical diagnosis.**
Upload only anonymized images you are authorized to share. Remove identifiers,
including text embedded in ultrasound pixels. The application does not retain
uploads after processing. Hugging Face operates the HTTPS proxy and hosting
infrastructure under its own terms and privacy policy.

The React dashboard and FastAPI service use frozen Custom CNN, ResNet18 and
post-hoc exploratory MobileNetV3-Small checkpoints. Probabilities are model
scores, not calibrated disease risks. Patient-independent and cross-centre
validation have not been established. Grad-CAM is a coarse sensitivity map,
not anatomical validation, segmentation or clinical evidence.

Source and full scientific limitations:
https://github.com/mweneh/testiculus-ultrasound-classification.
`SOURCE_REVISION.txt` identifies the deployed source revision. Trained weights
are stored separately in an approved **private** model repository, fetched only
by the server, pinned to a commit and verified against source-controlled hashes.
This Space does not offer checkpoint downloads. Unconfigured models are disabled.
No public deployment or approval is implied by this source template.

## Operator configuration

See `deployment/huggingface/README.md` in the GitHub source before deployment.
Set `DEPLOYMENT_MODE=space`, `PORT=7860`, `MODEL_REVISION` to the private model
repository's full commit SHA, and `MODEL_TYPES` to the desired model IDs.
Set `MODEL_REPO_ID` and a fine-grained, repository-read-only `HF_TOKEN` in Space
**secrets**. Never put tokens or checkpoints in this public repository or image.
The application verifies all package assets before loading selected models.
One worker, one active inference request and 12 admitted requests per minute
protect this shared CPU demonstration. A Space may sleep and need a cold start.

## Dataset attribution and component terms

Dataset: [TesticulUS](https://ditto.ing.unimore.it/testiculus/), Morelli et al.
and the TesticulUS contributors. Original images and adapted ultrasound images
(including Grad-CAM overlays) are subject to
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): retain
attribution, identify changes and follow applicable noncommercial/ShareAlike
terms when sharing adaptations. This does not apply that licence to unrelated
software or pretrained components; each retains its applicable terms.

Both required publications:

- Morelli et al., *Enhancing Testicular Ultrasound Image Classification Through
  Synthetic Data and Pretraining Strategies*, ICIAP 2025 proceedings, final
  publisher record (2026), pp. 26–39,
  https://doi.org/10.1007/978-3-032-10192-1_3.
- Morelli et al., *A New Multicenter Testicular US Dataset and a Lightweight
  Cond-UNet for Generalization in US Segmentation*, BMVC 2026. Provider-supplied
  BibTeX: https://github.com/mweneh/testiculus-ultrasound-classification/blob/main/data/references.bib.

See the source repository's `release/WEIGHT_NOTICES.md` for pretrained-weight
provenance and outstanding rights questions; private hosting requires an
approved hosting/usage review and is not a public redistribution permission.
Bundled frontend dependency and font notices are at `/THIRD_PARTY_NOTICES.txt`.
