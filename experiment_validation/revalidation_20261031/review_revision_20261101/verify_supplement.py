#!/usr/bin/env python3
"""Verify the S1 inventory and all 6600 retained lexical scores, without dependencies.

Run the copy at the root of an extracted supplement. No files are modified.
This checks retained evidence; it does not run models or certify the selector.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

BASE = 'experiment_validation/revalidation_20261031'
WORD = re.compile(r"[A-Za-z0-9']+")
LABELS = {'qwen_1_5b', 'qwen_3b', 'phi_1_5', 'llama_3_2_3b',
          'gemma_2_2b', 'tinyllama_1_1b', 'phi_3_5_mini'}
PILOT_LABELS = {'qwen_1_5b', 'llama_3_2_3b'}
CONDITIONS = {'baseline', 'bucket_8', 'pad_32', 'batch_2', 'batch_4', 'rand_pad_8'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,
            'Unsafe member path: ' + name)
    result = root.joinpath(*p.parts)
    require(result.resolve().is_relative_to(root.resolve()), 'Path escapes package: ' + name)
    return result


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def cosine(reference, prediction):
    a = Counter(WORD.findall(reference.lower()))
    b = Counter(WORD.findall(prediction.lower()))
    denominator = math.sqrt(sum(v * v for v in a.values()) *
                            sum(v * v for v in b.values()))
    return sum(v * b[k] for k, v in a.items()) / denominator if denominator else 0.0


def check(root):
    manifest_path = root / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(manifest['schema'] == 'lleakm-supplement-manifest/1', 'Unknown manifest schema')
    members = manifest['files']
    names = [x['path'] for x in members]
    require(len(names) == len(set(names)), 'Duplicate manifest path')
    require('MANIFEST.json' not in names and 'SHA256SUMS' not in names,
            'Manifest must exclude its two self-referential inventories')
    hashes = {}
    for item in members:
        path = safe_path(root, item['path'])
        require(path.is_file() and not path.is_symlink(), 'Missing/nonregular file: ' + item['path'])
        require(path.stat().st_size == item['bytes'], 'Size mismatch: ' + item['path'])
        digest = sha(path)
        require(digest == item['sha256'], 'Hash mismatch: ' + item['path'])
        hashes[item['path']] = digest
    checksum_rows = (root / 'SHA256SUMS').read_text(encoding='utf-8').splitlines()
    checksums = {}
    for line in checksum_rows:
        digest, name = line.split('  ', 1)
        require(name not in checksums, 'Duplicate checksum path: ' + name)
        checksums[name] = digest
    require(checksums == {**hashes, 'MANIFEST.json': sha(manifest_path)},
            'SHA256SUMS does not bind the manifest and all inventoried files')

    log_cache = {}
    checked_logs = set()
    fresh_ids = 0

    def verify_source(row, fresh=False):
        nonlocal fresh_ids
        name = row['generation_log_path']
        require(name in hashes, 'Source log absent from inventory: ' + name)
        require(hashes[name] == row['generation_log_sha256'], 'Bound log hash mismatch: ' + name)
        log = log_cache.get(name)
        if log is None:
            log = json.loads(safe_path(root, name).read_text(encoding='utf-8'))
            # The cache stores only reused pilot logs, not all large source files.
            if name.startswith('logs/') and row.get('victim_budget') == 96:
                log_cache[name] = log
        require(log['run_id'] == row['run_id'], 'Source run identity mismatch')
        require(log['response_text'] == row['reference'], 'Source reference mismatch')
        reference_hash = hashlib.sha256(row['reference'].encode('utf-8')).hexdigest()
        require(reference_hash == row['reference_sha256'], 'Reference hash mismatch')
        steps = log['steps']
        lengths = [s['token_utf8_len'] for s in steps]
        target = row.get('token_lengths', row.get('clean_lengths'))
        require(lengths == target, 'Logged source length vector mismatch')
        require(''.join(s['token_text'] for s in steps) == row['reference'],
                'Fragments do not concatenate to the reference')
        require(all(len(s['token_text'].encode('utf-8')) == s['token_utf8_len']
                    and s['sse_frame_utf8_len'] == s['token_utf8_len'] + 8 for s in steps),
                'Invalid UTF-8 or serialized-write lengths')
        if fresh:
            require(row['generated_token_ids'] == log['generated_token_ids'],
                    'Fresh generated ID mismatch')
            require(row['generated_token_ids'] == [s['token_id'] for s in steps],
                    'Fresh step IDs mismatch')
            fresh_ids += 1
        checked_logs.add(name)

    cases = [
        ('cuda_full', 'collection/canonical_traces.jsonl', LABELS),
        ('fresh_tinyllama_cuda', 'fresh_references/tinyllama/canonical_traces.jsonl', {'tinyllama_1_1b'}),
        ('fresh_phi_cuda', 'fresh_references/phi/canonical_traces.jsonl', {'phi_3_5_mini'}),
    ]
    total = 0
    maximum_error = 0.0
    counts = {}
    for run, inventory, labels in cases:
        inputs = list(rows(root / BASE / inventory))
        source_by_key = {(x['label'], x['idx']): x for x in inputs}
        expected = {(label, i) for label in labels for i in range(1, 301)}
        require(len(source_by_key) == len(inputs) and set(source_by_key) == expected,
                'Incomplete or duplicate canonical input keys: ' + run)
        for item in inputs:
            verify_source(item, fresh=run != 'cuda_full')
        seen = set()
        for row in rows(root / BASE / 'inference' / run / 'samples.jsonl'):
            key = row['label'], row['idx']
            require(key not in seen and key in expected, 'Unexpected/duplicate reconstruction key: ' + run)
            seen.add(key)
            source = source_by_key[key]
            for field in ['reference', 'token_lengths', 'reconstruction_seed', 'run_id',
                          'generation_log_sha256', 'reference_sha256']:
                require(row[field] == source[field], f'Input binding mismatch: {run}/{key}/{field}')
            calculated = cosine(row['reference'], row['prediction'])
            difference = abs(calculated - row['phi'])
            require(math.isfinite(row['phi']) and difference <= 1e-12, 'Lexical score mismatch')
            maximum_error = max(maximum_error, difference)
            total += 1
        require(seen == expected, 'Missing reconstruction keys: ' + run)
        counts[run] = len(seen)

    secondary = root / BASE / 'secondary/cuda_full'
    inventory = list(rows(secondary / 'input_inventory.jsonl'))
    secondary_inputs = {x['task_id']: x for x in inventory}
    require(len(secondary_inputs) == len(inventory) == 3900, 'Invalid secondary inventory')
    expected_secondary = {
        ('trace_control', label, i, condition, 'frozen', 20260706)
        for label in PILOT_LABELS for i in range(1, 31)
        for condition in ['real', 'shuffled', 'constant', 'cross_prompt']
    } | {
        ('defense_pilot', label, i, condition, 'frozen', 20260706)
        for label in PILOT_LABELS for i in range(1, 31) for condition in CONDITIONS
    } | {
        ('map', label, i, condition, attacker, seed)
        for label in PILOT_LABELS for i in range(1, 31) for condition in CONDITIONS
        for attacker in (['frozen'] if condition == 'baseline' else ['frozen', 'adaptive'])
        for seed in range(20260706, 20260711)
    }
    seen = set()
    seen_task_ids = set()
    secondary_counts = Counter()
    for row in rows(secondary / 'samples.jsonl'):
        key = tuple(row[k] for k in ['suite', 'label', 'idx', 'condition', 'attacker', 'seed_index'])
        require(key not in seen and key in expected_secondary, 'Unexpected/duplicate secondary key')
        seen.add(key)
        require(row['task_id'] not in seen_task_ids, 'Duplicate secondary task ID')
        seen_task_ids.add(row['task_id'])
        require(row['task_id'] in secondary_inputs, 'Secondary task missing from input inventory')
        item = secondary_inputs[row['task_id']]
        for name, value in item.items():
            require(name in row and row[name] == value, 'Secondary input binding mismatch: ' + name)
        source = dict(row, generation_log_path='logs/' + row['run_id'] + '.json')
        if source['generation_log_path'] not in checked_logs:
            verify_source(source)
        calculated = cosine(row['reference'], row['prediction'])
        recorded = row['metrics']['phi']
        difference = abs(calculated - recorded)
        require(math.isfinite(recorded) and difference <= 1e-12, 'Secondary lexical score mismatch')
        maximum_error = max(maximum_error, difference)
        secondary_counts[row['suite']] += 1
        total += 1
    require(seen == expected_secondary and seen_task_ids == set(secondary_inputs),
            'Incomplete secondary condition grid')
    require(total == 6600 and fresh_ids == 600, 'Incomplete total reconstruction/generation inventory')
    require(secondary_counts == {'trace_control': 240, 'defense_pilot': 360, 'map': 3300},
            'Unexpected secondary counts')
    return {
        'status': 'PASS', 'package_id': manifest['package_id'], 'files_hashed': len(members),
        'reconstruction_counts': {**counts, **dict(secondary_counts)},
        'reconstructions_checked': total, 'fresh_token_id_records_checked': fresh_ids,
        'source_logs_checked': len(checked_logs), 'maximum_phi_difference': maximum_error,
        'scope': 'Inventory, completeness, input/log bindings and lexical-score arithmetic. '
                 'No model inference, semantic disclosure assessment or selector certification.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        report = check(args.root.resolve())
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
