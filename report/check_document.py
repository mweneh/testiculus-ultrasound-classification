"""Check the report source, bibliography and compiled document without model access."""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
source = (HERE / 'capstone_report.tex').read_text()
bib = (HERE / 'references.bib').read_text()
data = json.loads((HERE / 'evidence.json').read_text())
abstract = source.split('% ABSTRACT_BEGIN')[1].split('% ABSTRACT_END')[0]
words = len(abstract.split())
assert 200 <= words <= 250, f'Abstract has {words} words'
assert bib.startswith((ROOT / 'data/references.bib').read_text()), 'Provider BibTeX changed'
keys = set(re.findall(r'@\w+\{([^,]+),', bib))
cited = {k for group in re.findall(r'\\cite[tp]?\{([^}]+)\}', source) for k in group.split(',')}
assert cited <= keys, f'Unknown citations: {cited - keys}'
assert {'2025ICIAP', '2026BMVC'} <= cited
labels = re.findall(r'\\label\{([^}]+)\}', source)
assert len(set(labels)) == len(labels), 'Duplicate label'
refs = set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}', source))
assert refs <= set(labels), f'Unknown cross-reference: {refs - set(labels)}'
assert all(label in refs for label in labels if label.startswith(('tab:', 'fig:')))
for model in data['models'].values():
    assert str(model['threshold']) in source, 'Threshold missing from manuscript'
    for partition in ['validation', 'validation_at_half']:
        values = model[partition]
        row = ' & '.join(f'{values[k]:.4f}' for k in
                         ['precision', 'recall_sensitivity', 'specificity', 'f1'])
        assert row in source, 'Manual validation operating-point table is stale'
for literal in ['191204', 'David Ongaro Mauti Ethan']:
    assert literal in source
pdf = HERE / 'capstone_report.pdf'
assert pdf.is_file() and pdf.stat().st_size > 100_000
pdf_text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
assert '??' not in pdf_text, 'Unresolved rendered reference'
normalized = ' '.join(pdf_text.split())
for literal in ['191204', 'David Ongaro Mauti Ethan', 'References',
                'Morelli', 'Cond-UNet', 'ShareAlike', '150', '0.7473', '0.5220']:
    assert literal in normalized, f'Expected text missing from PDF: {literal}'
for model in data['models'].values():
    assert str(model['threshold']) in normalized, 'Threshold absent from PDF'
log = (HERE / 'capstone_report.log').read_text()
assert 'Overfull' not in log, 'Rendered content exceeds its layout box'
assert 'undefined' not in log.lower(), 'Unresolved LaTeX reference or citation'
assert 'Rerun to get cross-references right' not in log
assert 'Warning--' not in (HERE / 'capstone_report.blg').read_text()
print(f'Document checks passed: {words}-word abstract, {len(cited)} citation keys, '
      f'{len(labels)} unique labels; provider bibliography unchanged; compiled PDF verified.')
