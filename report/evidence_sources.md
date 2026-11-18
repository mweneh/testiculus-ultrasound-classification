# Report evidence and unresolved information

The report describes the completed repository at development baseline `4dfd678`.
The course instructions were read from the supplied `DSA8401 Capstone Project.pdf`
(5 pages). They prescribe problem framing, technique justification, results,
limitations, reproducible scripts and API/Docker deployment; they provide no
report template or page limit. The PDF calls the submission individual. No
approved proposal was supplied. Student identity was supplied directly by the
student: David Ongaro Mauti Ethan, 191204. No supervisor identity is inferred.
The requested submission deadline is 20 November 2026; this report is prepared
on 18 November 2026.

## Numerical evidence

`evidence.json` is a report derivative, not a replacement for experimental
records. Its `source_sha256` mapping fingerprints every numerical source used
for report tables and plots. `build_evidence.py --collect` reads only saved
JSON/CSV records. It checks manifest counts, disjoint paths, validation/test
prediction identities, prediction hashes, all 129 recorded validation threshold
candidates per model, maximum-F1/tie rules, all test metrics and confusion
matrices, and best epochs against saved histories. It never imports model code,
opens ultrasound images, executes checkpoints, changes thresholds or writes
outside `report/`. The current task reads saved test scores for verification;
it does not perform test inference or optimize a test threshold.

| Claim or figure | Primary local evidence | Checks / interpretation |
|---|---|---|
| Class counts and image-level partitions | `data/splits/{train,validation,test}.csv` | 488/114, 104/25, 104/25; disjoint paths |
| Image dimensions and dataset integrity | `reports/dataset_audit.csv` (local), `reports/similarity_audit.md`, `data/README.md` | Prior audit; no new pixel audit in report preparation |
| Human similarity review | Final section of `reports/similarity_audit.md` | 22 reviewed, 12 cross-partition; no convincing cross-partition duplicates at contact-sheet resolution; no patient-independence inference |
| Architectures/preprocessing | `src/models/architectures.py`, `src/data/{dataset,transforms}.py` | Channel handling, normalization and classification heads read from implementation |
| Original hyperparameters | `reports/artifact_verification.json`, `training_runs/{custom_cnn,resnet18}/history.csv` | Checkpoint metadata confirms CUDA; original batch size not saved |
| MobileNet training | `reports/mobilenet_v3_small_freeze.json`, local `training_summary.json` and `history.csv` | CPU, frozen features/BatchNorm, head only; 10 epochs, best 5 |
| Thresholds and validation table | `reports/validation_thresholds.json`, `reports/mobilenet_v3_small_thresholds.json`, `results/*_validation/predictions.csv` | Validation-only selection; full sweep and prediction hashes preserved |
| Test table and confusion-matrix figure | `results/*_test/{metrics.json,predictions.csv}`, `reports/{artifact_verification,mobilenet_exploratory_results}.json` | Recomputed from stored scores only; identical 129 paths/labels |
| Training-loss figure | `training_runs/*/history.csv` | Raw recorded losses; saved checkpoint epochs marked |
| Efficiency table | `reports/model_efficiency.json`, `src/cpu_benchmark.py` | 50 raw measurements/model, 10 warm-ups; model-forward timing only |
| Application verification | `reports/artifact_verification.json`, `reports/mobilenet_application_verification.json` | Two validation samples/original model, one MobileNet validation threshold sample; synthetic Grad-CAM |
| Container verification | `reports/{container_verification,mobilenet_container_verification}.json`, `docs/mobilenet_experiment.md` | Synthetic uploads with trained checkpoints; cached build passed; fresh dependency download timed out |
| Test-suite count | `docs/mobilenet_experiment.md`, `README.md`, `AI_USE.md` | Latest previously verified suite: 150 passed, no deselections, one deprecation warning; report-only checks recorded separately |

## Literature and attribution

Both provider-supplied BibTeX entries are retained verbatim in `references.bib`
from `data/references.bib`. The official dataset page and original papers were
consulted on 18 November 2026. The supplied local `testiculus-real.pdf` states
that the release is fully anonymized and reports provider ethics approval;
this report attributes those statements to the provider and does not claim
separate institutional approval for this student project.

Primary sources checked:

- Dataset: <https://ditto.ing.unimore.it/testiculus/>.
- Morelli 2025 author manuscript: <https://federicobolelli.it/pub_files/2025iciap.pdf>;
  institutional record: <https://iris.unimore.it/handle/11380/1381588>.
- Morelli 2026 author manuscript (also supplied locally):
  <https://federicobolelli.it/media/publications/pdfs/0475.pdf>.
- ResNet: <https://arxiv.org/abs/1512.03385>.
- MobileNetV3: <https://arxiv.org/abs/1905.02244>.
- Medical transfer learning: <https://arxiv.org/abs/1902.07208>.
- AdamW: <https://arxiv.org/abs/1711.05101>.
- Grad-CAM: <https://arxiv.org/abs/1610.02391>.
- Precision–recall under imbalance:
  <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432>.
- Dataset licence: <https://creativecommons.org/licenses/by-nc-sa/4.0/>.

## Gaps requiring explicit qualification

1. Patient/examination and per-image centre IDs are absent. The provider's
   multicentre collection does not make the local image split centre-held-out.
2. Local image count is 1,060 versus the provider's 1,053. No release hash or
   version mapping resolves this; no images were removed to force agreement.
3. Original checkpoint metadata omits batch size, loader workers, elapsed
   training time, GPU model and original runtime versions. Current CLI defaults
   are not evidence of the original run. ResNet records ImageNet initialization
   but not its original downloaded weight checksum/enum/runtime.
4. Both original validation metric JSONs record majority baseline 0.8093023
   (696/860), rather than the partition baseline 0.8062016 (104/129). Model
   metrics reproduce and threshold selection is unaffected; original files stay
   unchanged. The report uses the verified partition baseline.
5. Original-model threshold-before-test ordering is documented by the supplied
   Colab experiment record; there is no independently authenticated execution
   log. MobileNet has an explicit hash-bound freeze and evaluation receipt.
6. MobileNet was added after observing the original test outcomes. Its own
   threshold was validation-selected before its test run, but the reused test
   partition is not independent confirmation of the expanded comparison.
7. No original-ultrasound Grad-CAM images were found in `results/`, `artifacts/`
   or `reports/`. Recorded synthetic map tests establish technical operation,
   not anatomical faithfulness. No medical heatmap was created for this report.
8. No confidence intervals, paired statistical tests, calibration study,
   clinical utility study, external evaluation, GPU latency or energy benchmark
   is claimed. One seed and 25 positives per validation/test partition limit
   inference. Do not interpret rounded differences as significant.
9. No new clean-environment install or uncached Docker build was repeated for
   this writing task. Prior cached-image verification is the deployment evidence.
10. The supplied 2025 BibTeX says pages 1–12; Springer's published volume lists
    pages 26–39: <https://link.springer.com/book/10.1007/978-3-032-10192-1>.
    The requested supplied entry is preserved; the author should decide whether
    to update pagination against the final publisher record before submission.
11. README and the earlier AI_USE opening still said similarity review was
    pending despite the completed review in the audit. The report uses the
    completed audit; its addition updates the disclosure of that completed
    review. No underlying split or experimental record is changed.
12. Checkpoints remain excluded from Git. A reviewer needs the separately
    supplied Google Drive artifacts and dataset access to run the complete
    pipeline; the tracked report, tables and plots rebuild without them.
