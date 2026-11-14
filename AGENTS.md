# AGENTS.md

## Project Overview

**Project:** Deep Learning for Testicular Ultrasound Classification Under Cross-Center Distribution Shift

**Context:** MSc Applied Machine Learning capstone project.

**Deadline:** Approximately one week.

**Primary objective:** Develop a reproducible deep learning pipeline for binary classification of testicular ultrasound images.

**Models:**
- Custom Convolutional Neural Network (CNN).
- ResNet18 with transfer learning.

**Additional deliverables:**
- Model evaluation and comparison.
- Grad-CAM explainability.
- FastAPI inference service.
- Docker containerization.
- Reproducibility documentation.
- Scientific report.

The repository will be examined as part of the academic assessment. Prioritize correctness, reproducibility, maintainability, and clear documentation.

## 1. General Development Rules

1. Inspect existing code before making changes.
2. Preserve working functionality unless modification is necessary.
3. Prefer simple, readable implementations over unnecessary abstractions.
4. Follow established Python conventions and use descriptive names.
5. Avoid introducing dependencies without justification.
6. Do not fabricate experimental results, test outcomes, or performance claims.
7. Do not silently modify the research methodology.
8. Document important assumptions and limitations.
9. Prioritize completing the end-to-end application over optional enhancements.

## 2. Atomic Git Commits

Use atomic commits throughout development.

Each commit must represent one coherent, working change.

Follow Conventional Commits:

`type(scope): concise description`

Examples:

- `feat(data): implement reproducible dataset splitting`
- `test(data): verify partition integrity`
- `feat(training): add PyTorch training loop`
- `feat(eval): implement classification metrics`
- `feat(xai): add Grad-CAM explanations`
- `feat(api): implement prediction endpoint`
- `build(docker): containerize inference service`
- `docs: update reproducibility instructions`
- `fix(training): correct checkpoint restoration`

### Before Every Commit

1. Run relevant tests.
2. Review the Git diff.
3. Check `git status`.
4. Stage only files related to the current change.
5. Confirm that no raw medical images, secrets, virtual environments, or large model checkpoints are staged.
6. Create the commit only when the relevant checks pass.

Do not bundle unrelated changes into one commit.

Do not rewrite existing Git history.

Do not create artificial commits for previously completed work.

Do not push to GitHub unless explicitly authorized.

## 3. Scientific and Methodological Safeguards

The classification dataset contains:

- 696 homogeneous images (class 0).
- 164 inhomogeneous images (class 1).
- 860 labelled classification images in total.

The dataset also contains 200 additional images without classification labels.

### Critical Restrictions

- Use only verified classification labels for supervised classification.
- Do not infer patient identities from image filenames.
- Do not infer acquisition centres from filenames or scanner appearance.
- Do not claim patient-independent validation without verified patient identifiers.
- Do not claim centre-held-out validation without verified centre identifiers.
- Do not describe synthetic image perturbations as genuine cross-centre validation.
- Do not use the test partition for hyperparameter tuning or model selection.
- Calculate class weights using training labels only.
- Apply stochastic augmentation only to training images.
- Keep validation and test preprocessing deterministic.
- Fit any data-dependent preprocessing using training data only.

The current evaluation protocol is provisional because patient-level and centre-level metadata are unavailable.

Document this limitation explicitly.

## 4. Dataset Integrity

The primary classification dataset consists of grayscale PNG images.

Expected image dimensions: 300 × 300 pixels.

The positive class is `inhomogeneous` (label 1).

The negative class is `homogeneous` (label 0).

One image, `homogeneous/793.png`, has an empty segmentation mask. Retain the image for ordinary classification.

Exclude it only from experiments requiring a valid segmentation mask.

Avoid relying exclusively on filename stems as unique identifiers. Use class-qualified relative paths where necessary.

Before model training, verify:

- Dataset integrity.
- Label consistency.
- Split reproducibility.
- Absence of overlapping images across partitions.
- Potential exact or near-duplicate images.

## 5. Model Development

Maintain two primary classification models:

1. Custom CNN trained from scratch.
2. ImageNet-pretrained ResNet18.

Use appropriate preprocessing for each model.

Prefer reusable training and evaluation functions rather than duplicating code.

Training should support:

- CPU and GPU execution.
- Fixed random seeds.
- AdamW