# Independent-audit resolution and submission readiness

Reviewed 19 November 2026 against audited revision `9e978eb` and the supplied
DSA 8401 assessment instructions. The [original audit](INDEPENDENT_AUDIT.md) is
preserved unchanged. Scientific experiments remain frozen.

| Finding | Status | Evidence/action |
|---|---|---|
| F1: GitHub source behind local submission | Resolved for the original submission | Authorized normal push published `b3567af`; GitHub API/tree and branch SHA verification confirmed all 98 files. The subsequent browser interface is integrated in merge `e5b84d6`; final documentation accompanies it on `main`. |
| F2: Examiner checkpoint access | Package verified; delivery pending | All 11 assets and three CPU checkpoint loads verified. No release exists. Rights review and explicit publication authorization remain required; fallback channel below is pending upload. |
| F3: Architecture diagram/layout | Resolved | README Mermaid flowchart separates development and inference, names all models, and documents Grad-CAM/API/Docker. Exact diagram rendered locally to SVG. |
| F4: Realistic complication | Resolved | Report §3.1 links natural imbalance, absent grouping metadata, integrity and correlated-image concerns to the requirement and bounds the mitigations. |
| F5: Bibliography capitalization | Resolved | Protected BMVC/licence titles; rendered US, Cond-UNet, NonCommercial and ShareAlike checked. ICIAP year 2026, pp. 26–39 and DOI retained. |
| F6: Release citation consistency | Resolved | Release notes cite the final ICIAP article; original supplied BibTeX retained with its provisional metadata explicitly identified. Only notes/control-file hashes changed. |
| F7: Exploratory marker in efficiency table (optional) | Resolved | Caption identifies MobileNet as post-hoc exploratory; measurements unchanged. |
| F8: Explainability subtitle suggestion (optional) | Retained | Subtitle unchanged; synthetic-only verification and absence of anatomical evidence remain explicit throughout the report. |

## Verification and preserved evidence

- Four report unit checks passed, including exact saved-artifact/snapshot and
  generated-table agreement. No numeric snapshot, plot or experimental record
  was regenerated.
- `report/build_evidence.py --verify-local --check` passed. Document checks
  passed: 212-word abstract, 10 citation keys, 12 labels, all references resolved.
- pdfLaTeX/BibTeX produced 22 pages with no undefined references, citation
  warnings or overfull boxes. Revised methods, efficiency, references and
  reproducibility pages were rendered and visually inspected.
- Release/hygiene checks: **19 passed** (18 release tests plus one notebook
  hygiene test). All three checkpoints restored strictly on CPU with matching
  preprocessing, finite state and thresholds; forward execution and pretrained
  downloads are forbidden by the integration test. All ten checksum entries
  match the 11-file package.
- Current tracked paths contain no model binaries, original images/contact
  sheets, dataset archives, credentials or temporary tooling. The scoped scan
  of 125 outgoing baseline blobs found no high-confidence credential patterns
  or blobs over 5 MB. Largest baseline tracked file: report PDF, 398,068 bytes.
  Follow-up files were reviewed separately. This is not an exhaustive history
  security audit. Existing Git history was not rewritten.
- Splits, all threshold records, weights, test results, confusion matrices,
  efficiency measurements, aggregate figures and model code are unchanged.
  No medical pixels, training or held-out inference were used for this work.
  The original comparison and exploratory MobileNet extension remain distinct.

The report, API, Dockerfile, Grad-CAM, MobileNet implementation, documentation
and tests were published at `b3567af` and independently verified on GitHub.
The later browser interface has been reviewed and merged without rewriting
the seven frontend commits. Independent integration checks passed 176 backend
tests with two synthetic-training tests deselected, four report tests and
32 frontend tests. Docker/browser checks used synthetic inputs only, including
all-model and ResNet18-only configurations. See
[integration verification](frontend_integration_verification.json).
The Docker build reused cached dependencies; the earlier uncached-download
timeout remains a reproducibility limit.

**Later verification (20 November 2026).** After the deployment branch,
interface copy and motion updates, class-label explanation and demo-image
selection script were merged into `main`, the full `python -m pytest -q` run
had 211 passed (no deselections; includes both synthetic-training tests, the
four report checks and four demo-image tests), and 35 frontend tests and the
production build passed. Demo-image preparation re-scored seven held-out
test-partition images with the release checkpoints solely to confirm they match
the saved test predictions; no checkpoint, threshold or reported metric changed.
The counts above remain the records of those earlier runs.

## Source synchronization and release commands

**The source push was authorized and verified. No checkpoint upload, model
release/tag creation or model publication has been performed.** Subsequent
verified source updates use the authorized normal push:

```bash
git push origin main:main
```

If remote `main` changes before execution, fetch/review the divergence; do not
force push. Use `git status --short --branch` and
`git rev-list --left-right --count origin/main...main` for current status.
Do not push ignored artifact directories or retarget a published tag.

[release/README.md](../release/README.md#commands-for-an-authorized-maintainer)
provides the complete explicit 11-asset `gh release create` command, draft review,
publication and anonymous-download/checksum/load checks. The versioned tag is
`models-v1.0.0`; its target must be the **final reviewed and pushed main commit**,
captured as `RELEASE_COMMIT=$(git rev-parse main)`. The older manifest field
`experiment_source_revision` is experiment provenance, not the tag target.
Before creating the release, compare the remote branch SHA to that full commit;
after publication compare the tag's resolved commit and verify all public
downloads. Do not claim a public release URL until those checks succeed.

After each source update, compare local and remote SHAs and verify the hosted
files. Source verification at `b3567af` is complete. After verified model
publication, update the README
and report to the exact versioned release URL, refresh the evidence/status
records and rebuild the PDF. These post-publication edits require no experiment
changes.

## Examiner access and owner decisions

Preferred route: GitHub source plus GitHub Release `models-v1.0.0`, with the
repository URL submitted through Virtual Campus. Source publication is complete;
checkpoint redistribution review and actual release publication remain pending.
Public checkpoint access is not claimed.

The owner-confirmed alternative is **Strathmore University Virtual Campus —
DSA 8401, Capstone Project (23 Nov) assignment submission**. Supply the matching
source checkout and complete `artifacts/releases/models-v1.0.0/` package through
that channel. **Upload and examiner-access verification are pending.** The
owner-supplied assignment label does not change the report's recorded
20 November deadline. Naming a channel does not demonstrate examiner access;
F2 remains open until checkpoint delivery is completed and checked.

Before publication or alternative delivery, the owner must resolve authority
to distribute trained checkpoints, appropriate checkpoint licence/notices,
applicable TesticulUS ShareAlike treatment of parameter files and ImageNet-derived
pretrained-weight terms. [WEIGHT_NOTICES.md](../release/WEIGHT_NOTICES.md) records
the sources and uncertainty. Torchvision's BSD software licence does not alone
settle pretrained-weight rights; ImageNet terms retrieval again timed out.
The software also has no repository-wide licence selected. Dataset CC BY-NC-SA
4.0 attribution and both mandatory citations are preserved. No new licence
grant or owner approval is inferred here.
