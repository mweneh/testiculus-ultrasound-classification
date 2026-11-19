# Independent audit — DSA 8401 capstone report

**Audited revision:** `9e978eb` (`docs(report): finalize submission reproducibility and references`). The working tree was clean at audit time.
**Audit date:** 19 November 2026. **Submission deadline:** 20 November 2026.
**Scope:** `report/capstone_report.tex` and `.pdf` (22 pages), `report/references.bib`, `report/evidence_sources.md`, `report/verification.json`, the course brief (`DSA8401 Capstone Project.pdf`), README, AI_USE.md, `release/`, `reports/`, saved `results/*/predictions.csv` and `metrics.json`, `training_runs/*/history.csv`, `data/splits/`, the relevant `src/` modules, git state, and the provider manuscript (`testiculus-real.pdf`).

**What this audit did not do:** No code, report, experiment, checkpoint, threshold or split was changed. No model was loaded or executed, no test inference was run, and no test suite was rerun. All recomputation used saved prediction CSVs, manifests and JSON records. The only external query was one Crossref DOI lookup.

---

## 1. Summary

The scientific content holds up. Every reported number I checked reproduces exactly from the primary artifacts. I found no data leakage in the implementation or the records. The report keeps MobileNetV3-Small framed as post-hoc and exploratory, states the Grad-CAM limitation accurately, and does not claim that the checkpoints are publicly available. The 150-test and 4-test results are reported as historical and report-specific records, not as runs during the writing task.

The main risks are about **examiner access**, not report content:

- the public GitHub repository is 23 commits behind the local branch and lacks the API, Docker and report deliverables
- the trained checkpoints are not reachable by any channel an examiner can use from the repository

There are also a few small compliance and citation-rendering gaps.

---

## 2. Independent verification performed

| Check | Method | Result |
|---|---|---|
| Split counts and disjointness | Read `data/splits/*.csv` | 602/129/129. Positives 114/25/25. Zero overlap between any pair of partitions ✔ |
| Prediction files match manifests | Compared paths and labels per model and partition | All 6 files match their manifest exactly ✔ |
| Test metrics (Table 5) | Recomputed AUC, AP, accuracy, balanced accuracy, precision, sensitivity, specificity and F1 at the frozen thresholds | All 24 values match to 4 dp. Stored test predictions equal `p ≥ t` ✔ |
| Confusion counts (§4.2, §5.1) | Same recomputation | CNN 10 TP/35 FP. ResNet 16/37. MobileNet 9/7, 106 vs 104 correct. "30 fewer FP, 7 fewer TP" ✔ |
| Validation operating points (Table 4) | Recomputed at 0.5 and at the selected threshold | All values match. MobileNet keeps 10 TP and FP falls from 9 to 6 ✔ |
| Threshold selection | Independent F1 sweep over unique validation scores, highest-threshold tie-break | Reproduces all three thresholds exactly. 129 unique candidates each ✔ |
| Class weights (§3.5) | `N/(2n_c)` on the training manifest | 0.616803 / 2.640351 ✔ |
| Histories (Table 2, §4.1) | Read `training_runs/*/history.csv` | Epochs 15/14/10. Best epochs 8/7/5. Best validation losses match ✔ |
| Efficiency (Table 7, §4.4) | Read `reports/model_efficiency.json` | Medians, means, p95, parameter counts and MiB sizes match. Ratios 5.9× and 7.2× ✔ |
| Validation baseline discrepancy (§4.3) | Read `results/*_validation/metrics.json` | Original files record 0.8093 (= 696/860). Partition value is 0.8062. The report describes this correctly ✔ |
| MobileNet ordering | Protocol, freeze and receipt timestamps against git log | Protocol 09:43 UTC → freeze 09:55 → single test evaluation 09:56–09:56:30, attempt 1. All records state that the original test outcomes had already been observed ✔ |
| Leakage safeguards in code | `src/training.py`, `src/data/transforms.py` | Class weights come from training labels. Early stopping and the scheduler use validation loss. Augmentation is training-only. Normalisation uses fixed ImageNet constants ✔ |
| Report integrity | SHA-256 of `.tex`, `.pdf`, `.bib` and `evidence.json` against `report/verification.json` | All four match. The PDF has 22 pages ✔ |
| ICIAP citation | Crossref lookup of `10.1007/978-3-032-10192-1_3` | Title, six authors, pages 26–39, 2026 and *ICIAP 2025* (LNCS) all match the bibliography ✔ |
| Provider facts (§3.1, §5.6) | `testiculus-real.pdf` | 1,053 images, two institutions, "fully anonymized" and ethics approval (CE AVEN 386/2025/OSS/AOUMO) match the report's provider-reported statements ✔ |
| Release status | grep of report, README, `release/`, `docs/` and the release notes | Every statement says "prepared, not published". No claim of public availability ✔ |
| Test-count framing | Report §4.6 and Appendix B, `verification.json`, AI_USE.md | 150 is labelled a historical application-suite result. 4 is labelled report-specific. `rerun_during_report_preparation: false` ✔ |

**Context on uncertainty (not a finding).** I ran a paired bootstrap on the saved test CSVs (4,000 resamples) and computed Wilson intervals:

- ResNet18 AUC 0.747, 95% CI about 0.63–0.85.
- ResNet18 minus Custom CNN AUC: about +0.01 to +0.26.
- ResNet18 minus MobileNet AUC: about −0.02 to +0.21, so the interval spans zero.
- Sensitivity at 16/25: [0.45, 0.80]. At 9/25: [0.20, 0.56].

This supports the report's choice to describe the differences as "observed" and descriptive and to avoid ranking by significance (§4.3, §5.1). No correction is required.

---

## 3. Findings

Severity scale:

- **Blocking**: must be resolved for the submission to be assessable.
- **Major**: likely to cost marks or mislead.
- **Minor**: should be fixed if time allows.
- **Low**: optional.

### F1 — Public repository does not contain the submission (Blocking for examiner access)

- **Location:** git state. Report Appendix A, `capstone_report.tex:412`. README line 21.
- **Evidence:**
  - `git status -sb` shows `main...origin/main [ahead 23]`.
  - `origin/main` is at `6d0a196` (17 Nov).
  - `git ls-tree origin/main` contains none of the following: `src/api.py`, `Dockerfile`, `src/gradcam.py`, `report/`, any MobileNet code, or the frozen result records.
  - The course brief (§4) requires "a single Git repository containing all the following", including the API, Docker and report.
  - The report itself directs examiners to the GitHub URL "pending an authorized push".
- **Impact:** An examiner cloning the stated URL receives an incomplete repository with no deployment and no report. The rubric's Deployment component (10%) and the report component cannot be assessed from it.
- **Smallest correction:**
  1. The owner authorises a normal, non-force push of `main`. No history rewrite is needed.
  2. Then update the one sentence at `tex:412` ("Examiners should use the matching supplied source checkout pending an authorized push…") so it no longer says the push is pending.
  3. Rebuild the PDF.

  If the submission is instead an uploaded archive or checkout, state that channel in the same sentence.

### F2 — Trained checkpoints are not reachable by examiners (Major)

- **Location:** `capstone_report.tex:412–431`. README lines 501–511. `release/README.md`. `.gitignore` lines 21–23.
- **Evidence:**
  - `artifacts/`, `training_runs/` and `results/` are git-ignored.
  - Release `models-v1.0.0` is unpublished. The report states this accurately at `tex:412`, and the redistribution review in `release/WEIGHT_NOTICES.md` is still open.
  - The brief requires "serialized artifacts for your chosen approach", and the rubric asks that deployment be "reproducible from the README".
- **Impact:** This is not a false claim; the report is honest. In practice, though, nobody outside this machine can run the API or container with trained weights. The per-image prediction CSVs are also outside Git, so `build_evidence.py --verify-local` cannot run on a clean clone. The committed `report/evidence.json` partly mitigates this.
- **Smallest correction:** Do one of the following before the deadline:
  - (a) Complete the owner redistribution review and publish `models-v1.0.0`. Then update the "not found … not claimed" sentence at `tex:412` and README lines 503–508.
  - (b) Deliver `artifacts/releases/models-v1.0.0/` to examiners through the official submission channel and name that channel in one sentence at `tex:412` and in the README.

  Do not state publication until it has happened.

### F3 — README has no architecture diagram (Minor, compliance)

- **Location:** `README.md`. The closest content is the directory tree at lines 877–899.
- **Evidence:** The deliverables checklist (§4) lists "README.md — project overview, setup instructions, **architecture diagram**, how to reproduce". A search for diagram, mermaid or flow constructs found none. The tree at lines 879–899 is a file layout, not a system diagram. Line 882 also still describes `models/` as "custom CNN and ResNet18 definitions", although `src/models/architectures.py:75` defines `build_mobilenet_v3_small`.
- **Smallest correction:**
  - Add a short Mermaid or ASCII diagram to the README showing: raw images → split manifests → training (validation-loss checkpoint) → validation threshold selection → frozen checkpoint and threshold → inference module → FastAPI → Docker, with read-only checkpoint mount.
  - Add "MobileNetV3-Small" to line 882.

### F4 — Course "realistic complication" requirement is not explicitly addressed (Minor, compliance/framing)

- **Location:** `capstone_report.tex:125` and `:88`.
- **Evidence:** The brief (§2) says that for a public dataset "you must introduce a realistic complication (e.g. … class imbalance) and justify it". The report says the imbalance "was not artificially introduced" (line 125). It also says the planned cross-centre design was abandoned (line 88). The substance is defensible: the dataset is a semi-raw research release with natural imbalance, missing grouping metadata and a provenance discrepancy. However, the report never connects this to the course requirement, so an examiner could read line 125 as non-compliance.
- **Smallest correction:** Add one sentence at line 125 explaining how the project meets the requirement. For example: the study treats the natural 4.24:1 imbalance, unverifiable patient/centre grouping (and the leakage audit it required) and the archive-count discrepancy as the realistic complications, and justifies each in §3.1–3.2. This is a framing change, not a new analysis.

### F5 — BMVC and CC titles are lowercased by `plainnat` (Minor, citation accuracy)

- **Location:** `report/references.bib` (entries `2026BMVC` and `cclicence`). Rendered in `capstone_report.bbl` lines 51–53.
- **Evidence:** The PDF bibliography prints "A new multicenter testicular **us** dataset and a lightweight **cond-unet** for generalization in **us** segmentation". "US" (ultrasound) and "Cond-UNet" are corrupted. The licence title renders as "Attribution--noncommercial--sharealike". The ICIAP title is already brace-protected.
- **Smallest correction:** Wrap both `title` fields in an extra pair of braces in `report/references.bib` and recompile. This changes no content.

### F6 — Superseded ICIAP record ships in the prepared release (Low, citation consistency)

- **Location:** `data/references.bib` and `artifacts/releases/models-v1.0.0/DATASET_REFERENCES.bib` (year 2025, pages 1–12, no DOI), compared with `report/references.bib` (2026, 26–39, DOI).
- **Evidence:** This is intentional and disclosed at `tex:435` and README lines 137–139. In-text citations render as "Morelli et al., 2026a" next to "ICIAP 2025". The appendix explains why, so this is consistent. The only residual risk is that the release, once published, would distribute the superseded pagination.
- **Smallest correction:** None required for the report. Optionally, add one line to the release notes before publication pointing to the version of record (DOI `10.1007/978-3-032-10192-1_3`, pp. 26–39).

### F7 — Efficiency table is the only MobileNet table without the exploratory marker (Low)

- **Location:** `report/generated/efficiency.tex` and the caption at `capstone_report.tex:311`.
- **Evidence:** Tables 3, 4, 5 and 6 and both figures mark MobileNet with `*` (post-hoc exploratory). Table 7 does not. The surrounding text (§4.4, §5.3) does not overstate. Measured latency is also not test-set-dependent, so the exploratory status matters less here.
- **Smallest correction:** Optional. Add "MobileNetV3-Small is the post-hoc exploratory model" to the Table 7 caption.

### F8 — Subtitle foregrounds "Explainability" more strongly than the evidence supports (Low)

- **Location:** `capstone_report.tex:38`.
- **Evidence:** The subtitle reads "Predictive Performance, Explainability, and Computational Efficiency". The body accurately limits explainability to a technically verified tool exercised on synthetic inputs, with no ultrasound attributions (§1.2 line 85, §4.5, abstract, conclusion). The course brief does not require explainability, so the missing ultrasound examples are not a compliance gap and the limitation is communicated accurately.
- **Smallest correction:** Optional. A subtitle such as "…, Explainability Tooling, and…" would align it with the body.

---

## 4. Items checked with no finding

- **Data leakage:**
  - No test data was used for checkpoint selection, threshold selection or class weights.
  - Validation reuse for both checkpoint and threshold selection is disclosed as optimistic (`tex:224`, `:346`).
  - Image-level splitting and possible within-examination correlation are disclosed repeatedly (title page, §3.2, §5.4, §5.5).
  - The similarity audit (22 candidate pairs, 12 cross-partition) is reported with its limits.
- **MobileNet exploratory status:**
  - Stated in the abstract, §1.1, §1.2 RQ3, §2.3, §2.5, Table 2, Table 3, Table 5, both figures, §5.4 and the conclusion.
  - §5.4 correctly separates within-experiment safeguards from the post-hoc decision to add the model.
  - No sentence treats MobileNet as confirmatory or superior.
- **Release:** No section claims public availability. Lines 412 and 431 explicitly condition the downloader on future publication.
- **Tests:** No section presents the 150-pass result as a run during report preparation (`tex:325`, `:455`; `verification.json`).
- **Grad-CAM:** Synthetic-input verification is described as technical only, and no anatomical claim is made (`tex:320–322`).
- **Unchanged artifacts:**
  - Hashes in `reports/mobilenet_exploratory_protocol.json` and `reports/mobilenet_v3_small_freeze.json` match the checkpoints and records used in the report.
  - The checkpoint SHA-256 values in `model_efficiency.json` match the protocol's original hashes.
  - The working tree is clean.
- **Technique justification:** Inclusion and exclusion reasoning is present (§2.5), as the brief §3.2 requires.
- **docker-compose:** The checklist accepts "or equivalent". The README `docker run` commands with read-only mounts (lines 729 and 748) are a reasonable equivalent. No correction is needed, though adding one README sentence saying so would remove any doubt.

Not independently verified: the characterisation of the ICIAP paper's method ("ResNet-based supervised and unsupervised pretraining, including diffusion-generated ultrasound data", `tex:94`). The paper's full text was not available locally. The title and Crossref metadata are consistent with that description.

---

## 5. Verdict

**Ready subject to minor corrections.** The report's scientific content, numbers, leakage handling, post-hoc framing and literature use need only minor correction.

This verdict depends on F1 and F2, which concern **access, not content**. If the repository is not pushed (F1) and checkpoints are not delivered by some channel (F2), examiners cannot assess the deployment component. In that case the submission should be treated as **not submission-ready**, however good the report is.

## 6. Minimum changes required before 20 November 2026

1. **F1 — Push `main` to `origin`** (owner authorisation, normal push, no force). Then revise the "pending an authorized push" sentence at `capstone_report.tex:412` and rebuild the PDF.
2. **F2 — Give examiners the checkpoints.** Either publish `models-v1.0.0` after the owner's redistribution review, or deliver the local package through the official submission channel. Update the one release-status sentence at `tex:412` and README lines 503–508 to match what actually happened. Do not claim publication in advance.
3. **F3 — Add a pipeline/architecture diagram to `README.md`** (a checklist item) and add MobileNet to the layout line 882.
4. **F5 — Brace-protect the BMVC and CC titles** in `report/references.bib` and recompile.

Recommended if time allows: F4 (one compliance-framing sentence). Optional: F6, F7, F8.

After any edit to the report, rerun the four report checks and the LaTeX build listed in Appendix B, and refresh `report/verification.json` hashes. None of these corrections requires retraining, threshold changes or test inference.
