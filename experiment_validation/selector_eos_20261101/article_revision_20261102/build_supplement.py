#!/usr/bin/env python3
"""Build a versioned local S2 EOS supplement after the manuscript is finalized.

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
BASE = ROOT / 'experiment_validation/selector_eos_20261101'
OLD = ROOT / 'experiment_validation/revalidation_20261031'
PACKAGE = 'LLeakM-S2-EOS-20261102-v1'
OMIT_PARTS = {'__pycache__', 'render', 'build', 'supplement', '.git'}
OMIT_NAMES = {'revision_manifest.json', 'completed_manifest.json', 'final_snapshot.json',
              'before_manifest.json', 'supplement_plan.json', 'supplement_validation.json',
              'supplement_preparation_check.json', 'revision_manifest.json', 'corrected_vs_original.json',
              'apply_article_revision.py', 'article_changes.json', 'article.diff',
              'ancillary.diff', 'prose_findings.md', 'packaging.log'}


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
            if item.name in OMIT_NAMES or item.name.startswith('.') or item.suffix in {'.pyc', '.so', '.tmp'}:
                continue
            if suffixes is not None and item.suffix not in suffixes:
                continue
            add(item)

    def bindings(mapping, label):
        for name, expected in mapping.items():
            path = ROOT / name
            if file_sha(path) != expected:
                raise ValueError(label + ' frozen source changed: ' + name)
            add(path)

    for name in ['app.py', 'client.py', 'weiss_reconstruction.py', 'requirements.txt', 'LICENSE', 'CITATION.cff']:
        add(ROOT / name)
    for name in ['template.tex', 'template.pdf', 'references.bib', 'COVER_LETTER.md',
                 'SUBMISSION_SYSTEM.md', 'SUBMISSION_CHECKLIST.md']:
        add(ROOT / 'article1' / name)
    tree(ROOT / 'article1/Definitions')
    for stem in ['eos_model_results', 'eos_defense_tradeoff']:
        for suffix in ['.pdf', '.svg', '.png']:
            add(ROOT / 'article1/figures' / (stem + suffix))
    tree(ROOT / 'experiment_validation/prompts')
    tree(ROOT / 'experiment_validation/scripts', suffixes={'.py'})
    tree(ROOT / 'experiment_validation/tests', suffixes={'.py'})
    for name in ['models_publication.json', 'requirements-analysis.txt', 'README.md']:
        add(ROOT / 'experiment_validation' / name)

    # The only article-result samples are the four NEW EOS campaign outputs.
    for name in ['cuda_full', 'fresh_tinyllama_cuda', 'fresh_phi_cuda']:
        tree(BASE / 'inference' / name)
        tree(BASE / 'inference_validation' / name)
        protocol = json.loads((BASE / 'inference' / name / 'protocol.json').read_text())
        bindings(protocol['original_source_sha256'], name)
    for name in ['rerun_canonical.py', 'candidate_evidence.py']:
        add(BASE / 'inference' / name)
    add(BASE / 'inference_validation/verify_inference.py')
    tree(BASE / 'secondary/cuda_full')
    add(BASE / 'secondary/rerun_secondary.py')
    tree(BASE / 'validation', suffixes={'.py', '.cpp', '.json', '.jsonl', '.md'})
    for name in ['PROTOKOL.md', 'queue_plan.json', 'collect_completed.py',
                 'paired_rendering.py', 'run_queue.py', 'queue/status.json']:
        add(BASE / name)
    bindings(json.loads((BASE / 'secondary/cuda_full/sources.json').read_text()), 'secondary')
    bindings(json.loads((BASE / 'queue_plan.json').read_text())['source_sha256'], 'queue')

    # Frozen reference inputs and old records required by the NEW protocols.
    # Their old reconstruction scores are provenance only, never article evidence.
    for name in ['canonical_traces.jsonl', 'rebuild_canonical_traces.py', 'canonical_trace_audit.json']:
        add(OLD / 'collection' / name)
    for name in ['tinyllama', 'phi']:
        tree(OLD / 'fresh_references' / name)
    for name in ['generate_fresh.py', 'test_generate_fresh.py', 'models_pinned.json', 'tokenizer_checks.json']:
        add(OLD / 'fresh_references' / name)
    add(OLD / 'references/app_before.py')
    inventories = [OLD / 'collection/canonical_traces.jsonl'] + [
        OLD / 'fresh_references' / name / 'canonical_traces.jsonl' for name in ['tinyllama', 'phi']]
    source_runs = set()
    for path in inventories:
        for row in read_rows(path):
            source = ROOT / row['generation_log_path']
            if file_sha(source) != row['generation_log_sha256']:
                raise ValueError('Frozen canonical source changed: ' + str(source))
            add(source)
            source_runs.add(row.get('historical_run_dir', row['run_dir']))
    for row in read_rows(BASE / 'secondary/cuda_full/input_inventory.jsonl'):
        source_runs.add(row['run_dir'])
    for name in source_runs:
        directory = ROOT / 'experiment_validation/results' / name
        add(directory / 'samples.jsonl')
        if (directory / 'config.json').is_file():
            add(directory / 'config.json')
    add(OLD / 'review_revision_20261101/sources_proposal.json')

    # New plots, data, table generator, checks and final build log are selected
    # when the parent runs this builder after completing the manuscript.
    # Before-state, rendered pages and package-dependent inventories are excluded.
    tree(HERE, exclude={'before', 'probe', 'verification_probe', 'supplement_test'},
         suffixes={'.py', '.json', '.jsonl', '.md', '.tex', '.csv', '.txt', '.log'})
    return selected


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Version already exists with different bytes; use a new package ID: ' + str(path))
        return
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def build(plan_only=False, destination=None, verify_tokenizers=False):
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
        'version_date': '2026-11-02',
        'scope': 'All manuscript reconstruction results and figures derive solely from the completed '
                 '6600-record EOS-inclusive campaign. Older inputs and comparison fields are retained '
                 'only where required for immutable source provenance. '
                 'This local package has not been publicly archived.',
        'exclusions': ['model weights and caches', 'partial campaigns and probes', 'old figures and PDF copies',
                       'old review narratives', 'full third-party papers', 'old S1 supplement',
                       'later package-dependent revision and packaging validation inventories'],
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
            command = [sys.executable, str(extracted / PACKAGE / 'verify_supplement.py')]
            if verify_tokenizers:
                command.append('--tokenizers')
            result = subprocess.run(command, text=True, capture_output=True)
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
    parser.add_argument('--tokenizers', action='store_true',
                        help='Also verify all encoder/decoder IDs offline with tokenizers installed')
    args = parser.parse_args()
    build(args.plan, args.output_dir, args.tokenizers)


if __name__ == '__main__':
    main()
