# AI Use Disclosure

AI assistance prepared an opt-in public-inference safety profile on a deployment
branch: bounded request bodies and upload duration, global admission and rate
limits, restricted multipart parsing, hidden interactive API documentation,
sanitized model-execution errors and anonymized-upload instructions. The local
API remains available with its existing contract. This step passed 37 API/safety
tests and 34 frontend tests; the production frontend build passed. No checkpoint
transfer, public deployment, model training or held-out inference was performed.

AI assistance then implemented private Hugging Face package bootstrap, immutable
revision/privacy checks, source-manifest SHA-256 verification, credential-safe
HTTPS redirects, model-subset selection, a narrow offline Space source bundle,
and a local/hosted inference comparison helper. It consulted current official
platform documentation and recorded outstanding account, cost and hosting-rights
conditions. No plugin connection to Hugging Face was available; the local CLI
was not authenticated. No token was requested in chat or persisted by the runtime.

The CPU Docker profile was checked locally under a two-CPU quota and 2 GiB cap.
Three-model startup took 20.38 seconds and peak cgroup memory was 482.39 MiB.
API probabilities matched local CPU inference on the same verified training image,
`inhomogeneous/128.png` (TesticulUS, Morelli et al., CC BY-NC-SA 4.0), manually
reviewed for visible identifiers. No test image was opened. Additional synthetic
inputs checked spooled-upload cleanup and desktop/mobile browser prediction and
Grad-CAM. No anatomical or clinical interpretation was made. All temporary test
containers were removed. The broader suite passed 201 backend tests with two
synthetic optimizer tests excluded, 34 frontend tests and four report tests.
Build/cache scope, operational timings, hashes, actual local checks and pending
public checks are recorded in `deployment/huggingface/verification.json` and its
deployment guide. These smoke timings do not replace the frozen scientific CPU
benchmark. Checkpoints and scientific records/report were preserved. No source
push, checkpoint transfer or hosted deployment was performed; the public
deployment objective remains incomplete pending the stated approvals and live
verification.

AI assistance was used to inspect the repository and implement the data split
and similarity-audit tooling, training and evaluation utilities, tests, and
project documentation. It also integrated the user-provided Google Colab
validation threshold record, its verification tools and configuration-based
evaluation/inference. The original thresholds and complete experimental record
were preserved. It subsequently implemented the frozen test-result summaries,
FastAPI prediction service, Grad-CAM for both architectures, Docker packaging
and synthetic application tests. The assistant also helped draft this disclosure.

The AI-assisted code and split methodology require project-developer review.
The completed human review recorded in `reports/similarity_audit.md` inspected
all 22 perceptual-hash candidate pairs, including 12 across partitions, and
found no convincing cross-partition duplicates at the available resolution.
Two within-training pairs appeared near-identical with different annotations;
no manifest assignments changed. These observations do not establish patient
identity or independence. The exact release provenance of the local
image copy remains unverified. The official dataset is licensed under
CC BY-NC-SA 4.0, as documented in [data/README.md](data/README.md).

The developer reports that both research models completed training, validation
threshold selection and final held-out test evaluation in Google Colab. Those
experiments are frozen. No research model training, threshold optimization or
held-out image evaluation was performed during the initial two-model application
implementation. The existing synthetic-training test was excluded from those checks.
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
test. At that phase, the suite had 129 passing tests and one deselected test out of
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

At the developer's request, the assistant corrected the dataset licensing
documentation using the official dataset page and Creative Commons licence.
It preserved the provider's BibTeX entries for both required publications and
documented attribution and ShareAlike for dataset-based Grad-CAM images and
other adaptations, separately from software licensing. This documentation
update did not remove additional images, rewrite history or modify the frozen
research artifacts.

The developer subsequently authorized one MobileNetV3-Small exploratory
experiment after the original test outcomes were already known. The assistant
implemented its architecture, preprocessing, frozen-feature training (including
BatchNorm statistics), checkpoint integration, separate validation-threshold
record, API, Grad-CAM and Docker support. It ran a synthetic CPU preflight,
then trained the MobileNet head on the unchanged training manifest using
ImageNet initialization, AdamW, training-label class weights, seed 42, batch
size 16 and learning rate 0.0001. The actual device was CPU; early stopping
completed 10 epochs in 93.44 seconds and selected epoch 5 by validation loss.
No feature blocks were subsequently fine-tuned.

The assistant generated validation predictions, applied the unchanged maximum-F1
selection criterion and froze threshold 0.5038942694664001 with checkpoint and
source hashes. Only then did it run one MobileNet evaluation on the existing
test images. This authorized post-hoc evaluation is separate from the earlier
application checks that did not open test images. No test-based tuning followed.
The original two checkpoints, thresholds, raw artifacts and aggregate results
were verified unchanged. Full-precision evidence and limitations are in
`reports/mobilenet_exploratory_results.json` and the experiment guide.

CPU efficiency measurements used the same hardware, input size, batch size,
thread counts, warm-up and repeated-forward method for all three trained models.
Existing classifiers were loaded for inference only. The synthetic training
probe optimized a disposable MobileNet copy. The extension's complete unit suite
also includes fresh synthetic training tests, including the previously excluded
test; these do not retrain the original research checkpoints. The final complete
suite passed 150 tests with one Starlette TestClient deprecation warning and no
deselections. The assistant helped write the exploratory comparison and this
disclosure. Human scientific
review remains required; one reused test partition does not establish a
confirmatory, patient-independent, centre-independent or clinical result.

The trained MobileNet API was separately compared with local inference on the
validation sample defining the selected threshold. API and local probabilities
matched exactly; the saved two-thread validation probability differed by
1.19e-7 with one thread and the decision agreed. Synthetic Grad-CAM preserved
predictions and parameter gradients. A separate Docker image passed synthetic
API and Grad-CAM checks for all three trained checkpoints, mounted read-only;
its temporary container was removed. The successful build reused the verified
dependency cache after an uncached PyTorch wheel download timed out. No package
versions or original deployment image tag were changed.

No clinical conclusions were generated. The project developer remains
responsible for confirming data provenance, reviewing the split policy,
verifying the original checkpoints and validation predictions, interpreting
results, and approving the final scientific report.

## Scientific report preparation

AI assistance was used to read the supplied DSA 8401 course instructions,
inspect source code and saved experimental artifacts, consult primary
literature, and draft the scientific report in `report/capstone_report.tex`.
It generated aggregate confusion-matrix and loss-history figures from saved
records, built the bibliography and PDF, checked citations/cross-references,
and inspected rendered pages. The supplied student name and registration
number were used on the title page. No approved proposal or supervisor details
were available.

Report-only checks recomputed metrics from saved numeric prediction CSVs,
verified validation threshold sweeps and hashes against their existing records,
and checked manifests and training histories. Four report-specific automated
tests passed. No models were executed, no ultrasound pixels were opened, no
training or held-out inference was repeated, and no experimental records or
thresholds were changed. The report distinguishes the previously recorded
150-pass application suite from these new document/evidence checks; the full
application suite was not rerun during writing.

The report explicitly discloses MobileNet's post-hoc status, missing grouping
metadata, the local/published dataset count discrepancy, missing original run
metadata and the validation-baseline inconsistency. No archived ultrasound
Grad-CAM outputs were available for scientific interpretation. The supplied
dataset BibTeX entries are preserved, with a publisher-pagination discrepancy
flagged for author review. The student remains responsible for the final
argument, references, institutional requirements and submitted document.

## GitHub Release checkpoint packaging

The assistant prepared local GitHub Release assets and a pinned source manifest
for the three existing final checkpoints, plus byte-identical copies of both
frozen threshold configurations. It added deterministic copy/checksum tooling,
strict CPU checkpoint-load verification, an optional standard-library public
GitHub downloader, release notes, dataset citations and source-linked
redistribution notices. Tests use mocked downloads and temporary synthetic
byte fixtures; a separate test strictly loads the available final checkpoints
without running their forward methods or downloading initialization weights.

All 18 release-tooling tests passed. Original checkpoint hashes were checked
against the frozen verification/freeze/benchmark records; checkpoint and
preprocessing metadata, finite states, class mappings, parameter counts and
threshold lookup were verified. No training, model forward pass, ultrasound
image access, test evaluation or threshold change occurred during packaging.
The assistant documented historical Google Drive provenance separately from
the planned GitHub Release distribution channel. Binary checkpoints remain
ignored and outside ordinary Git commits.

The assistant checked primary Torchvision/software licence and dataset sources.
Torchvision warns that pretrained models can carry additional dataset-derived
terms. ImageNet terms pages could not be retrieved; the scope of checkpoint
licensing, dataset ShareAlike treatment and pretrained-weight redistribution
remains for owner review. No blanket checkpoint/software licence was invented.
No GitHub Release, remote tag or push was created. Live release downloading
cannot be verified until an authorized release exists; download success and
failure handling were tested with mocked HTTPS responses.

## Targeted submission revision (19 November 2026)

AI assistance checked the assessment instructions and made limited report edits
for the prepared GitHub checkpoint distribution, Python/checkpoint/Docker paths
and examiner inference workflow. A read-only GitHub query returned `release not
found` for `models-v1.0.0`; the report does not claim public download availability.
Springer's final chapter record verified pages 26–39, DOI and publication on
2 January 2026 for the ICIAP 2025 paper. The report bibliography was corrected;
original provider and release citation files were preserved. No archived
ultrasound Grad-CAM outputs were found, so the synthetic-only limitation remains.

The assistant recompiled and inspected the revised PDF and ran report checks
against saved numeric artifacts. It did not rewrite the scientific argument,
change results, execute models, retrain, repeat held-out evaluation, publish a
release or push commits. The student retains responsibility for final review
and the independent scientific audit.

## Independent-audit corrections (19 November 2026)

Following the supplied Claude Code audit, the assistant made limited
submission-readiness changes: a README architecture diagram, explicit framing
of existing realistic complications, bibliography capitalization protection,
an exploratory-model efficiency caption and updated release-note citation
metadata. It preserved the original audit and all frozen scientific records.
The student identified Virtual Campus's DSA 8401, Capstone Project (23 Nov)
assignment as the alternative checkpoint submission channel; upload/access
verification remains pending and GitHub Releases remains preferred.

The assistant fetched remote references and checked release status read-only,
reviewed tracked/outgoing files, ignore rules and high-confidence credential
patterns, and prepared commands for an authorized normal push and versioned
release targeting the final reviewed source commit. No remote push, tag,
upload or release publication occurred. Weight-redistribution decisions remain
with the owner; no licence was invented.

Mermaid CLI rendered the README diagram locally using temporary tools outside
the project's dependencies. The assistant strictly loaded all three trained
checkpoints on CPU, checked package hashes/preprocessing/frozen thresholds and
reran 18 release-tooling tests plus one repository-hygiene test (19 passed).
It ran the four report tests and saved-evidence checks, rebuilt the 22-page PDF,
checked rendered bibliography capitalization and visually inspected affected
pages. These checks did not run model forward passes, retrain, open ultrasound
pixels, repeat held-out inference or regenerate experimental records. The
150-test application-suite result remains historical. The student remains
responsible for scientific review, rights decisions and verified examiner access.

## Web interface (19 November 2026)

AI assistance designed and implemented the `frontend/` web interface (React,
Vite, Tailwind CSS), a narrowly scoped `POST /explain` route reusing the
existing Grad-CAM implementation, optional static serving of the compiled
interface from the existing FastAPI application, a two-stage Dockerfile and a
`compose.yaml` demonstration, with tests and README documentation. Existing
`/health` and `/predict` behaviour and status codes were preserved and are
covered by the unchanged tests.

The comparison section reads the tracked frozen records at build time rather
than retyping metrics; a test checks the displayed values against the report.
The interface presents the model output as an uncalibrated score for the
inhomogeneous class, not a probability of disease, keeps MobileNetV3-Small
labelled post-hoc exploratory, and states Grad-CAM's limits. No checkpoint,
threshold, split, training record or evaluation result was changed; no model
was trained and no held-out test image was processed. Browser verification used
one training-partition image. The full 182-test run (including four report tests), 32
frontend tests and the container acceptance check passed on 19 November 2026.
The student remains responsible for reviewing the interface wording and
deciding whether to merge or publish it.

## Independent frontend integration review (19 November 2026)

The assistant reviewed the seven frontend commits against `main`, including
API contracts, model/threshold lookup, upload handling, Grad-CAM cleanup and
locking, static serving, Docker mounts and outgoing files. Live inference uses
backend responses; comparison values come from frozen tracked records. The two
committed screenshots were inspected and contain no uploaded images.

The review added complete bundled dependency/font notices and corrected mobile
overflow with long upload names. It also clarified transient upload retention
and that summary cards combine test metrics with separately measured CPU/model
properties. The original seven commits were preserved in merge `e5b84d6`.

Independent checks passed: 176 backend tests with the two synthetic optimizer
tests deselected, four report checks and 32 frontend tests; a production build,
Docker build and headless-browser verification of actual prediction/Grad-CAM
responses for all three trained models on synthetic inputs. A ResNet18-only
container verified partial-checkpoint operation and unavailable-model errors.
The Docker build used cached dependencies; an uncached build is not claimed.
No optimizer ran, no ultrasound pixels or held-out images were opened, and no
experimental record, threshold or checkpoint changed. The report received a
limited deployment/status update and was rebuilt. The user's authorization
covers source integration only; model files remain private and the release is unpublished.

