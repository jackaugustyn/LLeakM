"""Verify the reviewed final build and install its exact PDF, without inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
tex = ROOT / 'article1/template.tex'
pdf = HERE / 'build/template.pdf'
base = (HERE / 'before/article1/template.tex').read_text()
current = tex.read_text()
num = json.loads((HERE / 'manuscript_numeric_validation.json').read_text())
science = json.loads((HERE / 'validation/revision_science_check.json').read_text())
visual = json.loads((HERE / 'validation/visual_review.json').read_text())
assert num['status'] == 'PASS' and num['manuscript_sha256'] == sha(tex)
assert science['manuscript_sha256'] == sha(tex)
assert not science['unresolved_scientific_edit_issues']
assert sha(pdf) in json.dumps(visual), 'Visual report does not bind this PDF'
for name, value in num['artifact_sha256'].items():
    assert sha(ROOT / name) == value, name
for name, value in science['source_sha256'].items():
    assert sha(ROOT / name) == value, name
assert sha(ROOT / 'weiss_reconstruction.py') == science['data_preservation']['original_weiss_code_sha256']
assert current == (HERE / 'template.rev1.tex').read_text()
log = (HERE / 'build/template.log').read_text()
errors = [s for s in log.splitlines() if re.search(r'^!|Overfull|Underfull|undefined references|LaTeX Error|Citation.*undefined|Reference.*undefined', s)]
assert not errors, errors
protected = ['Author', 'AuthorNames', 'address', 'corres', 'authorcontributions', 'funding',
             'institutionalreview', 'informedconsent', 'conflictsofinterest']
for macro in protected:
    def get(text):
        return next(s for s in text.splitlines() if s.startswith('\\' + macro + '{') or s.startswith('\\' + macro + '['))
    assert get(base) == get(current), macro
def citation_keys(text):
    return sorted({key for body in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', text) for key in body.split(',')})
assert citation_keys(base) == citation_keys(current)
doc = pymupdf.open(pdf)
captions = {}
for label in [f'Table {i}' for i in range(1, 13)] + ['Table A1'] + [f'Figure {i}' for i in range(1, 5)]:
    pages = [i + 1 for i, page in enumerate(doc) if re.search(r'(?m)^' + re.escape(label) + r'\.', page.get_text())]
    assert len(pages) == 1, (label, pages)
    captions[label] = pages[0]
shutil.copy2(pdf, ROOT / 'article1/template.pdf')
assert sha(pdf) == sha(ROOT / 'article1/template.pdf')
report = {
    'status': 'PASS', 'verified_utc': datetime.now(timezone.utc).isoformat(), 'pages': len(doc),
    'tables': len(num['row_counts']), 'figures': 4, 'checked_table_and_plot_values': num['numeric_fields_checked'],
    'latex_errors': errors, 'caption_pages': captions, 'protected_declarations_unchanged': protected,
    'citation_keys_unchanged': citation_keys(current),
    'abstract': 'Scoped wording edits only; the abstract was not regenerated.',
    'warnings': 'Six standard hyperref PDF-metadata warnings for mathematical phi in the abstract; no layout or undefined-reference warnings.',
    'scientific_scope': 'Original PAD-inclusive selector retained and disclosed; corrected full-response performance not claimed.',
    'package_scope': 'This verifies the final manuscript only. The frozen supplementary archive has a separate extraction/verification report.',
    'sha256': {str(p.relative_to(ROOT)): sha(p) for p in [tex, ROOT / 'article1/template.pdf', ROOT / 'article1/references.bib',
                 HERE / 'build/template.log', HERE / 'manuscript_numeric_validation.json',
                 HERE / 'validation/revision_science_check.json', HERE / 'validation/visual_review.json']},
}
(HERE / 'final_build_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ['status', 'pages', 'tables', 'figures', 'checked_table_and_plot_values']}, indent=2))
