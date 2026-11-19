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
provider = (ROOT / 'data/references.bib').read_text()
# Preserve supplied metadata except verified ICIAP fields and title-case protection.
expected_provider = (provider.replace('month={Sep}', 'month={January}', 1)
                     .replace('year={2025}', 'year={2026}', 1)
                     .replace('pages={1--12}', 'pages={26--39}', 1))
expected_provider = expected_provider.replace(
    '    pages={26--39},',
    '    pages={26--39},\n'
    '    doi={10.1007/978-3-032-10192-1_3},\n'
    '    url={https://link.springer.com/chapter/10.1007/978-3-032-10192-1_3},\n'
    '    note={ICIAP 2025 conference; version of record published 2 January 2026},', 1)
bmvc_title = ('A New Multicenter Testicular US Dataset and a Lightweight Cond-UNet '
              'for Generalization in US Segmentation')
expected_provider = expected_provider.replace('title={' + bmvc_title + '}',
                                              'title={{' + bmvc_title + '}}', 1)
assert bib.startswith(expected_provider), 'Unexpected change to mandatory dataset references'
assert 'title={{Attribution--NonCommercial--ShareAlike 4.0 International}}' in bib
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
manifest = json.loads((ROOT / 'release/manifest.json').read_text())
for model in manifest['models'].values():
    assert model['checkpoint_asset'] in source, 'Release checkpoint filename missing'
assert 'Python 3.12' in source
assert 'Public checkpoint download is therefore not claimed.' in source
assert 'An uncached PyTorch dependency download timed out' in source
for literal in ['191204', 'David Ongaro Mauti Ethan']:
    assert literal in source
pdf = HERE / 'capstone_report.pdf'
assert pdf.is_file() and pdf.stat().st_size > 100_000
pdf_text = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
assert '??' not in pdf_text, 'Unresolved rendered reference'
normalized = ' '.join(pdf_text.split())
for literal in ['191204', 'David Ongaro Mauti Ethan', 'References',
                'Morelli', 'Testicular US', 'Cond-UNet', 'US Segmentation',
                'NonCommercial', 'ShareAlike', '150', '0.7473', '0.5220']:
    assert literal in normalized, f'Expected text missing from PDF: {literal}'
for model in data['models'].values():
    assert str(model['threshold']) in normalized, 'Threshold absent from PDF'
log = (HERE / 'capstone_report.log').read_text()
assert 'Overfull' not in log, 'Rendered content exceeds its layout box'
assert 'undefined' not in log.lower(), 'Unresolved LaTeX reference or citation'
assert 'Rerun to get cross-references right' not in log
assert 'Warning--' not in (HERE / 'capstone_report.blg').read_text()
print(f'Document checks passed: {words}-word abstract, {len(cited)} citation keys, '
      f'{len(labels)} unique labels; publisher metadata and release instructions verified; '
      'compiled PDF verified.')
