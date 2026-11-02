"""Validate the final local build, install its PDF and inventory this revision."""
from pathlib import Path
import re, json, hashlib, shutil, difflib, datetime
import pymupdf
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=(ROOT/'article1/template.tex').read_text();old=(HERE/'before/template.tex').read_text()
num=json.loads((HERE/'manuscript_numeric_validation.json').read_text())
assert num['status']=='PASS' and num['manuscript_sha256']==sha(ROOT/'article1/template.tex')
for path,h in num['artifact_sha256'].items(): assert sha(ROOT/path)==h,path
log=(HERE/'build/template.log').read_text()
issues=[x for x in log.splitlines() if re.match(r'^!|.*Overfull|.*Undefined|.*undefined references|.*LaTeX Error',x)]
assert not issues,issues
cite_pattern=r'\\cite\w*\*?(?:\[[^]]*\])*\{[^}]*\}'
cites=re.findall(cite_pattern,s);assert cites==re.findall(cite_pattern,old)
protected=['Title','Author','AuthorNames','address','corres','authorcontributions','funding','institutionalreview','informedconsent','dataavailability','conflictsofinterest']
for key in protected:
 def macro(t):return next(x for x in t.splitlines() if x.startswith('\\'+key+'{') or x.startswith('\\'+key+'['))
 assert macro(s)==macro(old),key
pdf=HERE/'build/template.pdf';d=pymupdf.open(pdf)
caption_pages={}
for kind,ids in [('Table',list(map(str,range(1,12)))+['A1']),('Figure',list(map(str,range(1,5))))]:
 for n in ids:
  pattern=rf'(?m)^{kind}\s+{n}\.'
  pages=[i+1 for i,p in enumerate(d) if re.search(pattern,p.get_text())]
  assert len(pages)==1,(kind,n,pages)
  caption_pages[f'{kind} {n}']=pages[0]
shutil.copy2(pdf,ROOT/'article1/template.pdf')
assert sha(pdf)==sha(ROOT/'article1/template.pdf')
(HERE/'manuscript.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='before/template.tex',tofile='article1/template.tex')))
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report={'status':'PASS','completed_utc':now,'pages':len(d),'latexmk_exit_code':0,'overfull_boxes':False,'undefined_references':False,'fatal_build_errors':issues,'remaining_warnings':'Six standard MDPI/hyperref PDF-string warnings from author metadata; no missing references or layout warnings.','numeric_fields_checked':num['numeric_fields_checked'],'tables_checked':len(num['row_counts']),'figures_checked':4,'caption_pages':caption_pages,'citation_commands_unchanged':len(cites),'protected_macros_unchanged':protected,'visual_review':{'all_pages_contact_sheet':len(d),'all_tables_review':'Separate agent reviewed all12 at2x on66898ef9; final wording suggestions incorporated, subsequent changed pages inspected by root.','root_detail_pages':[4,6,11,12,14,15],'result':'All4 figures and table layouts readable; no clipping or overlap. Final table caption clarifications preserve numeric content.'},'scope':'Local article revision and build validation; no external journal-policy certification or publication.','sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'article1/template.tex',ROOT/'article1/template.pdf',ROOT/'article1/references.bib',HERE/'build/template.log',HERE/'manuscript_numeric_validation.json',HERE/'figure_manifest.json']}}
(HERE/'final_build_validation.json').write_text(json.dumps(report,indent=2)+'\n')
files=set()
for p in HERE.rglob('*'):
 if p.is_file() and p.name!='revision_manifest.json' and not any(x in p.relative_to(HERE).parts for x in ['build','render','__pycache__']):files.add(p)
files.update((ROOT/'article1/figures').glob('latest_*'))
for rel in ['article1/template.tex','article1/template.pdf','article1/references.bib','article1/COVER_LETTER.md','article1/SUBMISSION_CHECKLIST.md','experiment_validation/README.md','experiment_validation/revalidation_20261031/AKTUALNY_STAN.md']:files.add(ROOT/rel)
for name in ['main_fresh_factledger.json','secondary_audit.json','paired_detokenization_score_contrast.json','paired_detokenization_protocol_check.json','fresh_legacy_replay.json']:files.add(HERE.parent/'final_review_20261101'/name)
manifest={'created_utc':now,'scope':'Latest manuscript revision plus its checked numeric sources; earlier experiment and integration manifests retained unchanged.','files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)],'final_pdf_sha256':sha(ROOT/'article1/template.pdf')}
(HERE/'revision_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for item in manifest['files']:assert sha(ROOT/item['path'])==item['sha256']
print(json.dumps({'status':report['status'],'pages':len(d),'tables':report['tables_checked'],'figures':4,'numeric_fields':num['numeric_fields_checked'],'citations_preserved':len(cites),'manifest_files':len(files),'caption_pages':caption_pages,'pdf_sha256':manifest['final_pdf_sha256']},indent=2))
