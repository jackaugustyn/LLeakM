#!/usr/bin/env python3
"""Build a versioned local S1 supplement after the manuscript is finalized.

No downloads or uploads. --plan lists the scope without creating an archive.
Every source byte is checked again while writing to detect concurrent edits.
The output is immutable: an existing version must have exactly the same bytes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / 'experiment_validation/revalidation_20261031'
PACKAGE = 'LLeakM-S1-20261101-v1'
OMIT_PARTS = {'__pycache__', 'render', 'build', 'supplement', '.git'}
OMIT_NAMES = {'revision_manifest.json', 'completed_manifest.json', 'final_snapshot.json',
              'before_manifest.json', 'supplement_plan.json', 'supplement_validation.json'}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_files():
    selected = {}

    def add(path):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError('Required regular source file missing: ' + str(path))
        relative = path.relative_to(ROOT).as_posix()
        if any(x in OMIT_PARTS for x in Path(relative).parts):
            raise ValueError('Forbidden source path: ' + relative)
        selected[relative] = path

    def tree(path, *, exclude=(), suffixes=None):
        if not path.is_dir():
            raise ValueError('Required source directory missing: ' + str(path))
        for item in sorted(path.rglob('*')):
            if not item.is_file():
                continue
            parts = item.relative_to(path).parts
            if any(x in OMIT_PARTS or x in exclude for x in parts):
                continue
            if item.name in OMIT_NAMES or item.suffix in {'.pyc', '.so', '.tmp'}:
                continue
            if suffixes is not None and item.suffix not in suffixes:
                continue
            add(item)

    for name in ['app.py', 'client.py', 'weiss_reconstruction.py', 'requirements.txt', 'LICENSE', 'CITATION.cff']:
        add(ROOT / name)
    for name in ['template.tex', 'template.pdf', 'references.bib', 'COVER_LETTER.md',
                 'SUBMISSION_SYSTEM.md', 'SUBMISSION_CHECKLIST.md']:
        add(ROOT / 'article1' / name)
    tree(ROOT / 'article1/Definitions')
    for item in sorted((ROOT / 'article1/figures').glob('latest_*')):
        add(item)
    tree(ROOT / 'experiment_validation/prompts')
    tree(ROOT / 'experiment_validation/scripts', suffixes={'.py'})
    tree(ROOT / 'experiment_validation/tests', suffixes={'.py'})
    for name in ['models_publication.json', 'requirements-analysis.txt', 'README.md']:
        add(ROOT / 'experiment_validation' / name)

    # These are the four completed reconstruction runs, without probes or partial reruns.
    for name in ['cuda_full', 'fresh_tinyllama_cuda', 'fresh_phi_cuda']:
        tree(BASE / 'inference' / name)
        tree(BASE / 'inference_validation' / name)
    add(BASE / 'inference/rerun_canonical.py')
    for name in ['verify_inference.py', 'test_verifier.py', 'README.md']:
        add(BASE / 'inference_validation' / name)
    tree(BASE / 'secondary/cuda_full')
    add(BASE / 'secondary/rerun_secondary.py')
    tree(BASE / 'collection', suffixes={'.jsonl', '.json', '.py', '.md'})
    for name in ['tinyllama', 'phi']:
        tree(BASE / 'fresh_references' / name)
    for name in ['generate_fresh.py', 'test_generate_fresh.py', 'models_pinned.json', 'tokenizer_checks.json']:
        add(BASE / 'fresh_references' / name)
    add(BASE / 'references/app_before.py')
    for name in ['completed_results.json', 'AKTUALNY_STAN.md', 'WYNIKI_WZNOWIENIA.md']:
        add(BASE / name)

    # Exact source dependencies are included, and their recorded bindings are checked.
    source_bindings = json.loads((BASE / 'secondary/cuda_full/sources.json').read_text())
    for name, expected in source_bindings.items():
        path = ROOT / name
        if file_sha(path) != expected:
            raise ValueError('Frozen secondary source changed: ' + name)
        add(path)
    inventories = [BASE / 'collection/canonical_traces.jsonl'] + [
        BASE / 'fresh_references' / name / 'canonical_traces.jsonl' for name in ['tinyllama', 'phi']]
    source_runs = set()
    for path in inventories:
        for row in read_rows(path):
            source = ROOT / row['generation_log_path']
            if file_sha(source) != row['generation_log_sha256']:
                raise ValueError('Frozen canonical source changed: ' + str(source))
            add(source)
            source_runs.add(row.get('historical_run_dir', row['run_dir']))
    for name in source_runs:
        tree(ROOT / 'experiment_validation/results' / name)
    for row in read_rows(BASE / 'secondary/cuda_full/input_inventory.jsonl'):
        source_runs.add(row['run_dir'])
    for name in source_runs:
        tree(ROOT / 'experiment_validation/results' / name)

    # Completed metric evidence and its independent analysis code.
    audit_names = ['audit_main_fresh.py', 'independent_dp.cpp', 'main_fresh_factledger.json',
                   'secondary_audit.py', 'secondary_audit.json', 'secondary_audit_metrics.jsonl',
                   'secondary_audit.md', 'paired_detokenization_score_contrast.json',
                   'paired_detokenization_protocol_check.json', 'fresh_legacy_replay.py',
                   'fresh_legacy_replay.json', 'fresh_legacy_replay_records.jsonl',
                   'fresh_legacy_replay.md', 'WERYFIKACJA_KONCOWA.md']
    for name in audit_names:
        add(BASE / 'final_review_20261101' / name)
    figure_dir = BASE / 'article_revision_latest_20261101'
    for name in ['generate_figures.py', 'figure_data.json', 'figure_manifest.json']:
        add(figure_dir / name)
    for item in figure_dir.glob('*.csv'):
        add(item)

    # This revision's scoped patches, checks and bounded ranking audit. Old PDF copies,
    # renderings and the later package-dependent revision manifest are excluded.
    tree(HERE, exclude={'before'}, suffixes={'.py', '.json', '.jsonl', '.md', '.tex', '.diff', '.csv', '.txt', '.log'})
    # The small source snapshots permit exact replay of the revision patch.
    # Old PDFs and unrelated before-state documents are not distributed.
    add(HERE / 'before/article1/template.tex')
    add(HERE / 'before/article1/references.bib')
    review = ROOT / 'reviews/ars_electronics_20261101'
    for name in ['RECENZJA.md', 'phase2_synthesis/decision_letter.md',
                 'phase2_synthesis/revision_roadmap.json',
                 'phase1_panel/methodology_artifacts/check_candidate_ranking.py',
                 'phase1_panel/methodology_artifacts/candidate_ranking_check.json',
                 'phase1_panel/domain_artifacts/audit_domain_inputs.py',
                 'phase1_panel/domain_artifacts/pad32_effective_input_audit.json',
                 'phase1_panel/da_artifacts/evidence_checks.json']:
        add(review / name)
    return selected


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Version already exists with different bytes; use a new package ID: ' + str(path))
        return
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def build(plan_only=False, destination=None):
    files = select_files()
    generated = {
        'README.md': (HERE / 'supplement_README.md').read_bytes(),
        'verify_supplement.py': (HERE / 'verify_supplement.py').read_bytes(),
        'requirements-cpu.txt': b'# Optional analysis/plot environment used on Python 3.10.\n'
                                b'# The standard-library verifier requires none of these.\n'
                                b'numpy==1.22.0\nscipy==1.8.0\nmatplotlib==3.5.1\n',
    }
    entries = []
    for name, path in sorted(files.items()):
        entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': file_sha(path)})
    for name, data in sorted(generated.items()):
        entries.append({'path': name, 'bytes': len(data), 'sha256': sha_bytes(data)})
    entries.sort(key=lambda item: item['path'])
    if len({x['path'] for x in entries}) != len(entries):
        raise ValueError('Duplicate generated member')
    summary = {'package_id': PACKAGE, 'files': len(entries),
               'uncompressed_bytes': sum(x['bytes'] for x in entries),
               'source_logs': sum('/logs/' in x['path'] or x['path'].startswith('logs/') for x in entries),
               'reconstruction_records': 6600, 'fresh_generation_records': 600}
    if plan_only:
        print(json.dumps({'mode': 'plan_only', **summary}, indent=2))
        return
    manifest = {
        'schema': 'lleakm-supplement-manifest/1', 'package_id': PACKAGE,
        'version_date': '2026-11-01',
        'scope': 'Frozen local supplementary data and code for the literal completed pipeline; '
                 'main numerical results retain the executed selector. A separate bounded '
                 'candidate-ranking audit is included. Public archival is pending.',
        'exclusions': ['model weights and caches', 'partial main-campaign/probe inference', 'old PDF copies',
                       'full third-party papers', 'historical whole-workspace inventories',
                       'the later package-dependent revision manifest'],
        'inventory_rule': 'MANIFEST.json excludes itself and SHA256SUMS; SHA256SUMS binds '
                          'the manifest and every listed file, excluding itself.',
        'files': entries,
    }
    generated['MANIFEST.json'] = json_bytes(manifest)
    checksums = {item['path']: item['sha256'] for item in entries}
    checksums['MANIFEST.json'] = sha_bytes(generated['MANIFEST.json'])
    generated['SHA256SUMS'] = ''.join(f'{digest}  {name}\n' for name, digest in sorted(checksums.items())).encode()
    destination = destination or ROOT / 'article1/supplement'
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (PACKAGE + '.tar.gz')
    temp = destination / (PACKAGE + '.tar.gz.tmp')
    entry_by_name = {item['path']: item for item in entries}
    all_names = sorted(set(files) | set(generated))
    try:
        with temp.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as gz:
                with tarfile.open(fileobj=gz, mode='w', format=tarfile.PAX_FORMAT) as archive:
                    for name in all_names:
                        data = generated[name] if name in generated else files[name].read_bytes()
                        if name in entry_by_name:
                            expected = entry_by_name[name]
                            if len(data) != expected['bytes'] or sha_bytes(data) != expected['sha256']:
                                raise ValueError('Source changed during packaging: ' + name)
                        info = tarfile.TarInfo(PACKAGE + '/' + name)
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        archive.addfile(info, io.BytesIO(data))
        # The verifier runs against a separately extracted copy, not mutable sources.
        with tempfile.TemporaryDirectory(prefix='lleakm-supplement-verify-') as scratch:
            extracted = Path(scratch)
            with tarfile.open(temp, 'r:gz') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.startswith(PACKAGE + '/'):
                        raise ValueError('Unexpected archive member: ' + member.name)
                    out = extracted / member.name
                    if not out.resolve().is_relative_to(extracted.resolve()):
                        raise ValueError('Archive member escapes extraction root')
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, out.open('wb') as sink:
                        while block := source.read(1024 * 1024):
                            sink.write(block)
            result = subprocess.run([sys.executable, str(extracted / PACKAGE / 'verify_supplement.py')],
                                    text=True, capture_output=True)
            if result.returncode:
                raise ValueError('Extracted package verification failed: ' + result.stdout + result.stderr)
            verification = json.loads(result.stdout)
        if target.exists():
            if file_sha(target) != file_sha(temp):
                raise ValueError('Frozen package version differs; use a new package ID: ' + str(target))
            temp.unlink()
        else:
            temp.replace(target)
        write_once(destination / (PACKAGE + '.manifest.json'), generated['MANIFEST.json'])
        digest = file_sha(target)
        write_once(destination / (PACKAGE + '.tar.gz.sha256'), f'{digest}  {target.name}\n'.encode())
        write_once(destination / (PACKAGE + '.verification.json'), json_bytes(verification))
        print(json.dumps({'status': 'PASS', **summary, 'archive': str(target),
                          'compressed_bytes': target.stat().st_size, 'sha256': digest,
                          'verification': verification}, indent=2))
    finally:
        if temp.exists():
            temp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--output-dir', type=Path, help='Alternate local directory; no external publication')
    args = parser.parse_args()
    build(args.plan, args.output_dir)


if __name__ == '__main__':
    main()
