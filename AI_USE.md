# AI Use Disclosure

AI assistance was used to inspect the repository and implement the data split
and similarity-audit tooling, training and evaluation utilities, tests, and
project documentation. It also integrated the user-provided Google Colab
validation threshold record, its verification tools and configuration-based
evaluation/inference. The original thresholds and complete experimental record
were preserved. It subsequently implemented the frozen test-result summaries,
FastAPI prediction service, Grad-CAM for both architectures, Docker packaging
and synthetic application tests. The assistant also helped draft this disclosure.

The AI-assisted code requires project-developer review. Automated tests have
been run, but the generated split and similarity candidates still require
human methodological review. In particular, 22 perceptual-hash candidate pairs
are listed in `reports/similarity_audit.md`; their similarity does not establish duplicate
acquisition or patient identity. The exact provenance and reuse terms of the
local image copy also require confirmation.

The developer reports that both research models completed training, validation
threshold selection and final held-out test evaluation in Google Colab. Those
experiments are frozen. No research model training, threshold optimization or
held-out image evaluation was performed during this application implementation.
The existing synthetic-training test was excluded from the application checks.
Its exact name is
`tests/test_training.py::test_synthetic_training_and_checkpoint_roundtrip`;
it optimizes a newly initialized model for one epoch on random 16×16 tensors.
It does not use medical images or the frozen checkpoints and was deselected
to avoid optimization during application verification, not because it failed.
Grad-CAM computes temporary attribution gradients without updating weights.

The validation record and final test numbers were supplied by the developer.
The implementation cross-checks reported count-based metrics against confusion
matrices and keeps validation and test results separate. Original predictions,
histories and trained checkpoints were subsequently downloaded from Google Drive.
The assistant verified file hashes, the complete validation selection record,
all saved test metrics from prediction CSVs, identical test paths/labels,
checkpoint metadata and training histories. It checked trained-model CLI/API
agreement on two validation samples per model and Grad-CAM on synthetic images.
That artifact verification opened no test images or test manifest. Local
validation probabilities match Colab within 2.1e-7; the verification tolerance
was 1e-5.

The original validation metric JSONs contain a baseline discrepancy (696/860
instead of 104/129). It is documented without editing the originals; the frozen
threshold selections and final test artifacts use the correct partition baseline.
Aggregate evidence and supplied-file hashes are in
`reports/artifact_verification.json`. The original rounded test-summary snapshot
and complete threshold record remain unchanged. Original Colab package/runtime
versions and independent provider checksums remain unavailable. These checks do
not establish patient independence, centre independence or clinical validity.

The Docker image was built and tested with both trained checkpoints mounted
read-only. Synthetic-image API results matched local CPU inference exactly;
both Grad-CAM commands and health/non-root/read-only checks passed. The temporary
test container was stopped and removed. Aggregate evidence is recorded in
`reports/container_verification.json`. The earlier application suite run passed
128 tests with one deselected. The subsequent repository audit found saved
ultrasound displays and image grids embedded in the tracked exploration
notebook. The assistant cleared its saved outputs and execution counts without
executing the notebook or changing its source cells, and added a regression
test. The current suite has 129 passing tests and one deselected test out of
130 collected, with one Starlette TestClient deprecation warning. The trained
artifact and container acceptance checks are separate from this count.

The assistant audited tracked filenames, notebook outputs and common credential
signatures. The credential scan read tracked text, including the test split
manifest; it did not decode held-out images or perform held-out model inference.
The current tracked files contain no medical image displays,
checkpoint files or detected credentials. Earlier Git commits still retain
the embedded notebook displays; no history rewrite or push was performed.
The assistant corrected the documented Python requirement to 3.12 and identified
the official Torchvision ImageNet weight source and the developer's Google Drive
checkpoint locations. The original pretrained-file hash and Colab runtime
remain unverified.

The assistant repeated the serving installation in an isolated Python 3.12
environment. Dependency checks, both trained-model inference and Grad-CAM CLIs,
and a temporary localhost HTTP API passed using only synthetic images. The API
probabilities matched the CLIs exactly, and the server was stopped afterward.
The pre-existing development environment inherits unrelated system packages
with dependency conflicts; these do not occur in the clean serving environment
or the Docker image, whose dependency check also passed.

No clinical conclusions were generated. The project developer remains
responsible for confirming data provenance, reviewing the split policy,
verifying the original checkpoints and validation predictions, interpreting
results, and approving the final scientific report.
