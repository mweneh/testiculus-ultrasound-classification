# Scientific capstone report

- [Submission PDF](capstone_report.pdf).
- [LaTeX source](capstone_report.tex) and [bibliography](references.bib).
- [Evidence/source audit and unresolved information](evidence_sources.md).
- [Full-precision report snapshot and source hashes](evidence.json).
- [Aggregate figures](figures/) and [generated numerical tables](generated/).

The source follows the supplied DSA 8401 assessment instructions. It explains
methodological choices, the evidence and justified conclusions. The repository
README retains installation and application usage instructions. No institutional
report template or approved proposal was supplied. Title-page identity is
David Ongaro Mauti Ethan, registration 191204, as provided by the student.

## Rebuild the PDF

The document uses the existing TeX Live installation, without new dependencies.
Run from the repository root:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error capstone_report.tex
bibtex capstone_report
pdflatex -interaction=nonstopmode -halt-on-error capstone_report.tex
pdflatex -interaction=nonstopmode -halt-on-error capstone_report.tex
```

The build needs only tracked report files, not dataset images, checkpoints or
original numeric artifacts. Intermediate TeX files are ignored. Rebuilds can
change PDF metadata timestamps without changing scientific content.

## Regenerate or check evidence

Use the existing development environment. Regenerate tables and figures from
the tracked snapshot without loading any model or external artifact:

```bash
MPLCONFIGDIR=/tmp/capstone-matplotlib .venv/bin/python report/build_evidence.py
```

With the separately supplied original numeric CSVs available, audit every saved
metric, manifest and validation threshold sweep against the report snapshot:

```bash
OMP_NUM_THREADS=1 .venv/bin/python report/build_evidence.py --verify-local --check
OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s report -p 'test_*.py' -v
.venv/bin/python report/check_document.py
```

The local-artifact unit check skips when separately supplied records are absent.
The other three tests use tracked/synthetic numeric data. `check_document.py`
checks the compiled PDF, citation/label integrity, abstract length, thresholds,
manual validation table, publisher-corrected ICIAP metadata, unchanged BMVC
metadata with brace-protected title capitalization, licence-title capitalization
and release-package instructions.
`--collect` is only for deliberately rebuilding the report's numeric snapshot
from saved artifacts; it never writes back to experiments. No command here
opens ultrasound pixels, executes a model, retrains or performs test inference.

## Verification status and author review

The final PDF was compiled with pdfLaTeX/BibTeX and rendered with Poppler for
visual page inspection. Four report-specific automated tests passed. The
application's previously recorded full-suite result remains **150 passed**;
that suite was not executed again during writing. The initial report phase
collected 154 tests (150 plus these four); release tooling subsequently added
18 separately verified tests. No research result was changed. Source hashes
were rechecked after drafting and during submission finalization.

The report explicitly records limitations rather than filling missing data.
Before submission, the student should approve the argument, institutional
submission details and scope, confirm examiner access to separately supplied
checkpoints/artifacts. The ICIAP reference has been corrected to Springer's
version of record: pages 26–39, DOI `10.1007/978-3-032-10192-1_3`, publication
year 2026 for the ICIAP 2025 conference. Its key `2025ICIAP` and the mandatory
BMVC citation are retained; the original provider BibTeX is preserved, while
release notes now identify the final published ICIAP record.
The release query on 19 November 2026 returned `release not found`, so the
report describes planned distribution and local examiner access, without
claiming public downloads. Redistribution review and authorized publication
remain outstanding. See the evidence audit for remaining gaps.

The [independent audit](INDEPENDENT_AUDIT.md) is preserved unchanged. The
[submission-readiness record](submission_readiness.md) lists the corrections,
verification and remaining access decisions. The owner-confirmed fallback is
Strathmore University Virtual Campus — DSA 8401, **Capstone Project (23 Nov)**
assignment submission. Upload and examiner-access verification are pending;
GitHub Releases remains preferred. This assignment label does not revise the
report's recorded deadline.
