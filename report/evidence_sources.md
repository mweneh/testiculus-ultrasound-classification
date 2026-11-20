# Report evidence and unresolved information

The scientific experiments describe development baseline `4dfd678`. The targeted
submission revision on 19 November 2026 also incorporates the distribution
tooling/documentation committed in `38aa5d1`, without changing experiments.
The course instructions were read from the supplied `DSA8401 Capstone Project.pdf`
(5 pages). They prescribe problem framing, technique justification, results,
limitations, reproducible scripts and API/Docker deployment; they provide no
report template or page limit. The PDF calls the submission individual. No
approved proposal was supplied. Student identity was supplied directly by the
student: David Ongaro Mauti Ethan, 191204. No supervisor identity is inferred.
The requested submission deadline is 20 November 2026; the report was initially
prepared on 18 November and revised on 19 November 2026.

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
| Checkpoint distribution | `release/{manifest,verification}.json`, `release/SHA256SUMS`, `release/README.md`; local package | All three source checkpoint fingerprints and all package assets rechecked; publication remains outstanding |
| Test-suite count | `docs/mobilenet_experiment.md`, `README.md`, `AI_USE.md` | Latest previously verified suite: 150 passed, no deselections, one deprecation warning; report-only checks recorded separately |

## Literature and attribution

The report's ICIAP reference now follows the final publisher record, verified on
19 November 2026: pages 26–39; DOI `10.1007/978-3-032-10192-1_3`; first online
2 January 2026. ICIAP 2025 is the conference year; 2026 is the version-of-record
publication year, so the report's author–year citations now distinguish the two
2026 Morelli papers. The BibTeX key `2025ICIAP` is retained. Only the report's
publication fields change; its authors/title and the BMVC entry are preserved.
The original provider entries remain byte-identical in `data/references.bib`
and the prepared release, preserving release hashes. The official dataset page
and original papers were consulted on 18 November 2026. The supplied local `testiculus-real.pdf` states
that the release is fully anonymized and reports provider ethics approval;
this report attributes those statements to the provider and does not claim
separate institutional approval for this student project.

Primary sources checked:

- Dataset: <https://ditto.ing.unimore.it/testiculus/>.
- Morelli 2025 author manuscript: <https://federicobolelli.it/pub_files/2025iciap.pdf>;
  institutional record: <https://iris.unimore.it/handle/11380/1381588>.
- Final ICIAP publisher record:
  <https://link.springer.com/chapter/10.1007/978-3-032-10192-1_3>.
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
10. The supplied ICIAP BibTeX's pagination discrepancy is resolved in the report:
    pages 26–39 and publication year 2026 follow the publisher chapter record.
    Original provider BibTeX entries remain historical records; the audit
    follow-up below records the release-note citation correction.
11. README and the earlier AI_USE opening said similarity review was pending
    despite the completed review in the audit. This documentation discrepancy
    is corrected alongside the report, using the completed audit. No
    underlying split or experimental record is changed.
12. Checkpoints remain excluded from Git. The planned distribution channel is
    GitHub Releases; the original two-model Google Drive transfer is historical
    provenance. Pending publication, an examiner needs the locally supplied
    package or original checkpoints. Dataset access is needed for full pipeline
    reproduction, but not a single prediction on a separately supplied image.
    The tracked report, tables and plots rebuild without either.

## Submission-finalization checks (19 November 2026)

- Read-only command `gh release view models-v1.0.0 --repo
  mweneh/testiculus-ultrasound-classification --json
  tagName,isDraft,publishedAt,url,assets` returned `release not found`. The
  initial Snap CLI invocation could not run inside the sandbox; the read-only
  retry outside the sandbox completed with that result. No release URL or
  public download availability is asserted in the report. Authorization to
  publish and owner resolution of weight-redistribution questions remain
  outstanding. Submission commits also require an authorized source push;
  the examiner workflow explicitly uses a matching supplied checkout until
  then. No upload, tag creation or push was performed.
- The local `artifacts/releases/models-v1.0.0/` package matches its trusted
  manifest, all asset hashes and both full threshold records. Original source
  checkpoint bytes also match the frozen release manifest. The package and
  its notices/citation files were not changed during this report revision.
- Path-name inspection under `artifacts/`, `results/` and `reports/` found no
  Grad-CAM/heatmap/overlay artifacts. Existing JSON evidence identifies only
  synthetic Grad-CAM verification. The report retains its limitation; no
  ultrasound image was opened or generated.
- Serving documentation and report use Python 3.12 (host verification 3.12.3),
  pinned `requirements-serving.txt`, CPU loading, the same model-specific
  threshold files, and checkpoint assets outside Git. Original checkpoints
  remain under `training_runs/`; the supplied package is under
  `artifacts/releases/`; future downloader installation is under
  `artifacts/checkpoints/`. All package paths use the pinned `models-v1.0.0` tag.
- README now supplies a Docker mount/environment mapping for flat release
  filenames. This is a static documentation update, not a new Docker run.
  The report's cached-build success and uncached-download timeout are unchanged.
- Scientific tables, figures, evidence snapshot, splits, model code, thresholds
  and original experimental records remain unchanged. Report checks read saved
  numeric artifacts only; no training, inference or test evaluation is repeated.
  Final compilation, visual inspection and test results are recorded in
  `verification.json`.

## Independent-audit follow-up (19 November 2026)

The complete supplied [independent audit](INDEPENDENT_AUDIT.md) is retained
unchanged (SHA-256 `fb6909c63aaabca41925a14156a7234103133d1af073e25348235da238ec7be1`).
Its saved-prediction bootstrap observations were not added to the frozen
scientific results. See [submission readiness](submission_readiness.md) for
remaining owner actions.

- **F3:** README now includes a Mermaid flowchart separating completed model
  development from deployed inference, names all three architectures and
  describes Docker's read-only checkpoint mount. The exact fenced block was
  rendered successfully to SVG with Mermaid CLI 11.15.0 in temporary tooling.
  It uses GitHub's documented `mermaid` fence and standard flowchart syntax:
  <https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams>.
  A GitHub-hosted rendering is not claimed before pushing.
- **F4:** Section 3.1 explicitly connects natural imbalance, absent grouping
  metadata, the archive-count discrepancy and possibly correlated images to
  the capstone's realistic-complication requirement. Class weighting,
  imbalance-aware evaluation and integrity/similarity checks address aspects
  of these complications without establishing patient/centre independence.
- **F5:** Brace protection preserves `US`, `Cond-UNet`, `NonCommercial` and
  `ShareAlike`. Rendered BibTeX/PDF output was checked. Springer again confirms
  the ICIAP proceedings paper's 2026 publication, pp. 26–39 and DOI; the mandatory
  BMVC metadata and CC BY-NC-SA 4.0 attribution remain intact.
- **F6:** The release notes now identify the ICIAP version of record. The
  provider's original BibTeX remains byte-identical. Only release notes,
  manifest and checksum metadata changed in the ignored 11-file package;
  original/packaged checkpoint and both threshold hashes remain unchanged.
  Strict CPU loads and preprocessing/threshold checks passed for all models,
  without forward passes or image access. The 18 release tests and one
  repository-hygiene test passed again (19 total); this is separate from the
  historical 150-test application record.
- **F7 (optional):** The efficiency-table caption now explicitly identifies
  MobileNetV3-Small as the post-hoc exploratory model. No numerical table or
  figure changed. The optional subtitle suggestion F8 was left unchanged;
  the existing synthetic-only explainability limitation remains explicit.
- **F1/F2 remain pending access actions:** A fresh `git fetch origin` confirmed
  `origin/main` at `6d0a196bbe702913d4f4a242e681d1ba0518f660`, with 23 local
  commits ahead and none behind before these corrections. The read-only
  release query again returned `release not found`. Git path/blob, notebook,
  ignore-rule and credential-pattern checks found no forbidden current tracked
  artifacts or high-confidence credentials in the outgoing blobs. These are
  scoped checks, not an exhaustive security audit. No push, remote tag, upload
  or publication was performed.
- The owner confirmed the official alternative: **Strathmore University Virtual
  Campus — DSA 8401, Capstone Project (23 Nov) assignment submission**. Upload
  and examiner-access verification remain pending. GitHub Releases is preferred,
  with the repository URL submitted through Virtual Campus. The report's
  20 November deadline remains unchanged; no delivery is implied by naming a
  channel.
- Primary Torchvision documentation still cautions that pretrained weights can
  have dataset-derived terms. Both ImageNet terms-page retrievals timed out
  again; checkpoint rights/ShareAlike treatment and ImageNet-derived
  redistribution remain owner decisions, not assumed permissions. Dataset and
  Creative Commons notices were rechecked; neither the dataset licence nor
  model-weight notices were changed.
- Four report tests and saved-evidence checks passed; the PDF was rebuilt with
  pdfLaTeX/BibTeX and affected pages visually inspected. No retraining,
  ultrasound-pixel access, test inference, threshold/split/result changes or
  experimental-record regeneration occurred. Verification hashes are refreshed
  in `verification.json`.

## Independent browser-interface integration (19 November 2026)

- Reviewed the original seven commits (`0860e2a` through `e7b2f26`) against
  `b3567af`. Existing prediction/health contracts, frozen thresholds, class
  mappings and preprocessing are preserved. The browser consumes actual
  `/health`, `/predict` and `/explain` responses; historical comparison values
  are imported from the frozen tracked JSONs at build time.
- `/explain` shares the 5 MiB/20-megapixel upload checks, uses a reentrant model
  lock around prediction and attribution, returns a bounded 224×224 heatmap
  and relies on the existing hook/mode cleanup. Invalid/unconfigured inputs
  produce the documented status codes. This is a local research service, not
  an audited public clinical service.
- Two review findings were corrected: `7a64eef` retains complete bundled font
  and production dependency notices; `1468b22` prevents grid overflow caused
  by a longer upload filename at 390 pixels. A headless-browser regression
  check verified the corrected layout and real backend responses on synthetic
  pixels. Merge `e5b84d6` preserves the seven original commits and both fixes.
- Independent checks: 176 backend tests passed, with the two fresh synthetic
  optimizer tests deselected; four report checks passed; 32 frontend tests and
  the production build passed. A Docker build and browser/API checks passed
  for three trained models; a ResNet18-only container also passed, with the
  unconfigured models returning 503. Dependency layers were cached; no fresh
  network build is claimed. Full results are in
  `frontend_integration_verification.json`.
- Later verification (20 November 2026), after merging the deployment branch,
  interface copy/motion updates, class-label explanation and demo-image script
  into `main`: full `python -m pytest -q` had 211 passed with no deselections
  (both synthetic-training tests, four report checks and four demo-image tests
  included); 35 frontend tests and the production build passed. Demo-image
  preparation re-scored seven held-out test-partition images with the release
  checkpoints only to confirm the saved test predictions; no checkpoint,
  threshold or reported metric changed. The 176/32 counts above are the records
  of the earlier integration run.
- The current integration opened no ultrasound pixels and ran no optimizer,
  held-out inference or scientific evaluation. Earlier frontend work separately
  documented one training-image demonstration, with private screenshots under
  ignored artifacts. Those outputs were not used as anatomical evidence or
  included in the report. The two tracked UI screenshots were inspected and
  contain no uploaded image or medical pixels.
- Outgoing paths/blobs were reviewed for credentials, raw images, checkpoints,
  archives and large files. Only the two explicitly inspected, image-free
  UI screenshots are permitted raster additions. This is a scoped check, not
  an exhaustive security audit. Original experimental records, both threshold
  configurations, checkpoints, splits, figures and numerical tables are
  unchanged.
- Source publication of `b3567af` was independently verified by GitHub API and
  tree equality. The report/README now distinguish published source from the
  still-unpublished checkpoint release. Redistribution review remains open;
  this integration does not publish weights or modify release-package bytes.
  The original independent audit is retained unchanged. PDF compilation,
  visual inspection and hashes are refreshed in `verification.json`.
