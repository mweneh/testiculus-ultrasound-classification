# Independent-audit resolution and submission readiness

Reviewed 19 November 2026 against audited revision `9e978eb` and the supplied
DSA 8401 assessment instructions. The [original audit](INDEPENDENT_AUDIT.md) is
preserved unchanged. Scientific experiments remain frozen.

| Finding | Status | Evidence/action |
|---|---|---|
| F1: GitHub source behind local submission | Prepared; owner authorization pending | Fresh fetch found `origin/main` at `6d0a196`; no divergence. Before corrections, local `main` was 23 commits ahead. New correction commits also remain local. Normal push only; verify remote deliverables afterwards. |
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

The current source checkout contains the report, API, Dockerfile, Grad-CAM,
MobileNet implementation, documentation and tests. This statement does not
assert that GitHub already contains the local deliverables. The historical
150-pass application suite was not rerun. No new Docker build is claimed;
cached-build evidence and the uncached dependency-download timeout remain the
recorded deployment limits.

## Source synchronization and release commands

**No push, upload, release/tag creation or publication has been performed.**
The exact normal push, after explicit owner authorization, is:

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

After an authorized push, verify that GitHub exposes the report/PDF, API,
Dockerfile, Grad-CAM, MobileNet code, README/AI_USE and tests at the pushed SHA.
Then update pending-source text. After verified publication, update the README
and report to the exact versioned release URL, refresh the evidence/status
records and rebuild the PDF. These post-publication edits require no experiment
changes.

## Examiner access and owner decisions

Preferred route: GitHub source plus GitHub Release `models-v1.0.0`, with the
repository URL submitted through Virtual Campus. Both source synchronization
and release publication remain pending authorization; public checkpoint access
is not claimed.

The owner-confirmed alternative is **Strathmore University Virtual Campus —
DSA 8401, Capstone Project (23 Nov) assignment submission**. Supply the matching
source checkout and complete `artifacts/releases/models-v1.0.0/` package through
that channel. **Upload and examiner-access verification are pending.** The
owner-supplied assignment label does not change the report's recorded
20 November deadline. Naming a channel does not demonstrate examiner access;
F1/F2 remain open until the selected delivery route is completed and checked.

Before publication or alternative delivery, the owner must resolve authority
to distribute trained checkpoints, appropriate checkpoint licence/notices,
applicable TesticulUS ShareAlike treatment of parameter files and ImageNet-derived
pretrained-weight terms. [WEIGHT_NOTICES.md](../release/WEIGHT_NOTICES.md) records
the sources and uncertainty. Torchvision's BSD software licence does not alone
settle pretrained-weight rights; ImageNet terms retrieval again timed out.
The software also has no repository-wide licence selected. Dataset CC BY-NC-SA
4.0 attribution and both mandatory citations are preserved. No new licence
grant or owner approval is inferred here.
