#!/usr/bin/env python3
"""Verify the S2 inventory and all 6600 NEW EOS-bounded reconstruction records.

Default: Python 3.10+ standard library, no downloads, no file modification.
--tokenizers additionally re-encodes all contexts and decodes every candidate
using the two bundled, hash-pinned tokenizer JSON files; no model weights.
Neither mode reproduces model forward passes or original victim generation.
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

BASE = 'experiment_validation/selector_eos_20261101'
OLD = 'experiment_validation/revalidation_20261031'
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


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def canonical_digest(value):
    return hashlib.sha256(packed(value).encode('utf-8')).hexdigest()


def numeric(value):
    if isinstance(value, str):
        require(value in {'NaN', 'Infinity', '-Infinity'}, 'Unknown nonfinite score encoding')
        return float(value)
    require(type(value) in {int, float}, 'Nonnumeric candidate score')
    return float(value)


def check_sum(values, recorded):
    numbers = [numeric(value) for value in values]
    require(all(math.isfinite(value) for value in numbers), 'Nonfinite EOS-prefix score')
    require(all(value <= 1e-6 for value in numbers), 'Positive normalized log score')
    expected = math.fsum(numbers)
    saved = numeric(recorded)
    # Original CUDA reductions use float32; this independent CPU sum uses float64.
    tolerance = max(1e-5, 2e-6 * max(1.0, sum(abs(value) for value in numbers)))
    require(math.isfinite(saved) and abs(expected - saved) <= tolerance,
            'EOS-inclusive prefix sum mismatch')


def split_segments(lengths):
    cuts = [-1] + [i for i, value in enumerate(lengths) if value == 1]
    if cuts[-1] != len(lengths) - 1:
        cuts.append(len(lengths) - 1)
    chunks = [lengths[a + 1:b + 1] for a, b in zip(cuts, cuts[1:])]
    for i in range(len(chunks) - 1):
        if chunks[i][-3:] == [3, 1, 1]:
            chunks[i + 1] = chunks[i][-3:] + chunks[i + 1]
            chunks[i] = chunks[i][:-3]
    result, pending = [], []
    for chunk in filter(None, chunks):
        pending += chunk
        if len(pending) >= 10:
            result.append(pending)
            pending = []
    if pending:
        result.append(pending)
    return result


def hf_cleanup(text):
    # Transformers 4.46.3 PreTrainedTokenizerBase.clean_up_tokenization;
    # clean_up_tokenization_spaces=True is pinned in both tokenizer configs.
    for old, new in [(' .', '.'), (' ?', '?'), (' !', '!'), (' ,', ','),
                     (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"),
                     (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")]:
        text = text.replace(old, new)
    return text


def load_tokenizers(root, hashes, enabled):
    directory = root / BASE / 'article_revision_20261102/pinned_tokenizers'
    metadata = json.loads((directory / 'SOURCES.json').read_text(encoding='utf-8'))
    protocol = json.loads((root / BASE / 'secondary/cuda_full/protocol.json').read_text())
    result = {}
    if enabled:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ValueError('--tokenizers requires the tokenizers package; default mode is standard-library only') from exc
    for model, entry in metadata.items():
        require(entry['revision'] == protocol['checkpoints'][model], 'Tokenizer revision mismatch')
        modeldir = directory / model.split('/')[1]
        for name, item in entry['files'].items():
            key = (modeldir / name).relative_to(root).as_posix()
            require(hashes.get(key) == item['sha256'] == protocol['checkpoint_files_sha256'][model][name],
                    'Pinned tokenizer/config hash mismatch')
        config = json.loads((modeldir / 'config.json').read_text())
        require(config['eos_token_id'] == 1 and config['pad_token_id'] == 0 and
                config['decoder_start_token_id'] == 0, 'Pinned T5 special token IDs changed')
        if enabled:
            full = Tokenizer.from_file(str(modeldir / 'tokenizer.json'))
            full.no_truncation()
            full.no_padding()
            kept = Tokenizer.from_file(str(modeldir / 'tokenizer.json'))
            kept.no_padding()
            kept.enable_truncation(max_length=80)
            tkconfig = json.loads((modeldir / 'tokenizer_config.json').read_text())
            require(tkconfig['clean_up_tokenization_spaces'] is True, 'Tokenizer cleanup setting changed')
            result[model] = (full, kept)
    require(set(metadata) == {'royweiss1/T5_FirstSentences', 'royweiss1/T5_MiddleSentences'},
            'Incomplete tokenizer metadata')
    return result


def check_candidates(row, tokenizers, *, secondary=False):
    require(row['record_sha256'] == canonical_digest({k: v for k, v in row.items() if k != 'record_sha256'}),
            'New reconstruction record digest mismatch')
    length_key = 'attacker_input_lengths' if secondary else 'token_lengths'
    segments = split_segments(row[length_key])
    max_segments = row['max_sentences'] if secondary else 0
    requested = min(max_segments, len(segments)) if max_segments else len(segments)
    samples = row['samples_per_segment'] if secondary else 3
    require(samples == (1 if secondary and row['suite'] == 'map' else 3), 'Incorrect candidate sampling regime')
    calls = row['actual_encoder_inputs']
    require(row['available_segments'] == len(segments), 'Incorrect available segment count')
    require(0 < len(calls) <= requested, 'Incorrect attempted segment count')
    counts = Counter(records=1)
    chosen = []
    for position, call in enumerate(calls):
        model = 'royweiss1/T5_FirstSentences' if position == 0 else 'royweiss1/T5_MiddleSentences'
        require(call['model_id'] == model and call['max_length'] == 80, 'Encoder model/limit mismatch')
        suffix = ''.join(' _' + str(value) for value in segments[position])
        expected = ('Translate the Special Tokens to English. \nSpecial Tokens:' + suffix if position == 0
                    else 'Translate the Special Tokens to English, given the context. \nContext: ' +
                    chosen[-1] + ' \nSpecial Tokens:' + suffix)
        require(call['text'] == expected, 'Length serialization/context chain mismatch')
        retained = call['retained_token_ids']
        require(0 < len(retained) <= 80 and all(type(value) is int and value >= 0 for value in retained),
                'Invalid retained encoder IDs')
        require(call['truncated'] == (call['full_token_count'] > len(retained)), 'Truncation flag mismatch')
        require(call['full_token_count'] >= len(retained), 'Full encoder token count too small')
        if 'full_token_ids' in call:
            full = call['full_token_ids']
            require(len(full) == call['full_token_count'] and all(type(value) is int and value >= 0 for value in full),
                    'Invalid full encoder IDs')
            if not call['truncated']:
                require(full == retained, 'Untruncated encoder IDs differ')
        if tokenizers:
            full_tokenizer, kept_tokenizer = tokenizers[model]
            full = full_tokenizer.encode(expected).ids
            kept = kept_tokenizer.encode(expected).ids
            require(retained == kept and call['full_token_count'] == len(full), 'Recomputed encoder IDs differ')
            if 'full_token_ids' in call:
                require(call['full_token_ids'] == full, 'Recomputed full encoder IDs differ')
            counts['encoder_calls_retokenized'] += 1
        audit = call['candidate_audit']
        require(audit['schema'] == 'eos-bounded-candidate-audit/1' and
                audit['selector'] == 'processed-score-sum-through-first-eos', 'Candidate schema/selector mismatch')
        require(audit['samples_requested'] == call['samples'] == samples, 'Candidate sampling count mismatch')
        require(audit['decoder_prefix_length'] == 1 and audit['eos_token_ids'] == [1], 'Decoder special token mismatch')
        require(audit['score_dtype'] == ('torch.float32' if samples > 1 else None), 'Score dtype mismatch')
        candidates = audit['candidates']
        require(len(candidates) == samples, 'Missing candidate evidence')
        counts.update(calls=1, candidates=len(candidates), ranked_calls=int(samples > 1),
                      single_candidate_calls=int(samples == 1))
        best = {}
        for index, candidate in enumerate(candidates):
            require(candidate['candidate_index'] == index, 'Candidate index mismatch')
            sequence = candidate['sequence_token_ids']
            generated = candidate['generated_token_ids']
            require(sequence and sequence[0] == 0 and len(sequence) <= 80 and sequence[1:] == generated,
                    'Decoder prefix/generated token alignment mismatch')
            require(all(type(value) is int and value >= 0 for value in sequence), 'Invalid decoder token ID')
            text = candidate['text']
            require(isinstance(text, str) and text == text.strip() and candidate['eligible'] == bool(text),
                    'Candidate text/eligibility mismatch')
            if tokenizers:
                decoded = hf_cleanup(full_tokenizer.decode(sequence, skip_special_tokens=True)).strip()
                require(decoded == text, 'Recomputed candidate decoding differs')
                counts['candidates_redecoded'] += 1
            eos = next((i for i, value in enumerate(generated) if value == 1), None)
            require(candidate['first_eos_generated_offset'] == eos, 'First EOS offset mismatch')
            if eos is not None:
                require(all(value == 0 for value in generated[eos + 1:]), 'Non-PAD token after first EOS')
            if samples == 1:
                require(candidate['transition_scores'] is None and candidate['confidence'] == 0 and
                        candidate['scored_token_count'] == 0 and candidate['legacy_full_step_confidence'] is None,
                        'Singleton candidate must bypass confidence ranking')
            else:
                scores = candidate['transition_scores']
                require(len(scores) == len(generated), 'Transition score/token count mismatch')
                n = eos + 1 if eos is not None else len(generated)
                require(candidate['scored_token_count'] == n, 'EOS-inclusive score count mismatch')
                check_sum(scores[:n], candidate['confidence'])
                counts['scored_tokens'] += n
                tail = [numeric(value) for value in scores[n:]]
                if tail:
                    counts['candidates_with_post_eos_tail'] += 1
                    counts['excluded_post_eos_tokens'] += len(tail)
                    counts['candidates_with_nonfinite_post_eos_tail'] += int(any(not math.isfinite(value) for value in tail))
            confidence = numeric(candidate['confidence'])
            require(math.isfinite(confidence), 'Nonfinite selected-rule confidence')
            if text and (text not in best or confidence > best[text][0]):
                # Text insertion order and first candidate on an exact tie are preserved.
                best[text] = (confidence, index)
        ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
        texts = [text for text, _ in ranked]
        indices = [value[1] for _, value in ranked]
        require(audit['ranked_texts'] == texts and audit['ranked_candidate_indices'] == indices,
                'Exact deduplicated candidate ranking mismatch')
        selected = texts[0] if texts else ''
        require(audit['selected_candidate_index'] == (indices[0] if indices else None) and
                call['selected_text'] == selected, 'Selected candidate mismatch')
        chosen.append(selected)
        if position > 0 and not selected:
            require(position == len(calls) - 1, 'Continued after an empty middle selection')
    require(len(calls) == requested or (len(calls) >= 2 and not chosen[-1]), 'Unexplained early stop')
    completed = 1 + sum(bool(text) for text in chosen[1:])
    prediction = re.sub(r'\s+([,.;:!?])', r'\1', ' '.join(text.strip() for text in chosen if text.strip()))
    require(row['reconstructed_segments'] == completed and row['first_prediction'] == chosen[0] and
            row['prediction'] == prediction, 'Final response assembly mismatch')
    if secondary and row['suite'] == 'map':
        require(len(calls) == 1 and samples == 1, 'MAP scope must be one segment and one candidate')
        counts['new_map_records_checked'] += 1
    return counts


def check(root, with_tokenizers=False):
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

    tokenizers = load_tokenizers(root, hashes, with_tokenizers)
    candidate_counts = Counter()
    bindings_checked = 0

    def verify_bindings(mapping):
        nonlocal bindings_checked
        for name, expected in mapping.items():
            require(hashes.get(name) == expected, 'Protocol source binding mismatch: ' + name)
            bindings_checked += 1

    queue = json.loads((root / BASE / 'queue_plan.json').read_text())
    verify_bindings(queue['source_sha256'])
    secondary_protocol = json.loads((root / BASE / 'secondary/cuda_full/protocol.json').read_text())
    secondary_sources = json.loads((root / BASE / 'secondary/cuda_full/sources.json').read_text())
    require(secondary_protocol['sources'] == secondary_sources, 'Secondary source inventories differ')
    verify_bindings(secondary_sources)
    require(secondary_protocol['selector_version'] == 'eos-through-first-inclusive/1', 'Wrong secondary selector')
    secondary_protocol_id = canonical_digest(secondary_protocol)
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
        inputs = list(rows(root / OLD / inventory))
        protocol = json.loads((root / BASE / 'inference' / run / 'protocol.json').read_text())
        verify_bindings(protocol['original_source_sha256'])
        require(protocol['selector_version'] == 'eos-through-first-inclusive/1', 'Wrong main selector')
        require(protocol['input_sha256'] == hashes[OLD + '/' + inventory], 'Canonical inventory hash mismatch')
        require(protocol['reconstructor_sha256'] == hashes['weiss_reconstruction.py'], 'Reconstructor hash mismatch')
        require(protocol['runner_sha256'] == hashes[BASE + '/inference/rerun_canonical.py'], 'Runner hash mismatch')
        protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
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
            require(row['protocol_id'] == protocol_id, 'Main record protocol mismatch')
            candidate_counts.update(check_candidates(row, tokenizers))
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
        require(row['protocol_id'] == secondary_protocol_id, 'Secondary record protocol mismatch')
        candidate_counts.update(check_candidates(row, tokenizers, secondary=True))
        item = secondary_inputs[row['task_id']]
        require(item['input_sha256'] == canonical_digest({k: v for k, v in item.items() if k != 'input_sha256'}),
                'Secondary input record digest mismatch')
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
    expected_candidate_counts = dict(records=6600, calls=86788, candidates=253764,
                                     ranked_calls=83488, single_candidate_calls=3300,
                                     scored_tokens=4540261, candidates_with_post_eos_tail=142629,
                                     excluded_post_eos_tokens=393353,
                                     candidates_with_nonfinite_post_eos_tail=142629,
                                     new_map_records_checked=3300)
    for key, value in expected_candidate_counts.items():
        require(candidate_counts[key] == value, 'Incomplete candidate audit count: ' + key)
    if with_tokenizers:
        require(candidate_counts['encoder_calls_retokenized'] == 86788 and
                candidate_counts['candidates_redecoded'] == 253764, 'Incomplete tokenizer audit')
    return {
        'status': 'PASS', 'package_id': manifest['package_id'], 'files_hashed': len(members),
        'reconstruction_counts': {**counts, **dict(secondary_counts)},
        'reconstructions_checked': total, 'fresh_token_id_records_checked': fresh_ids,
        'source_logs_checked': len(checked_logs), 'maximum_phi_difference': maximum_error,
        'protocol_source_bindings_checked': bindings_checked,
        'candidate_evidence': dict(candidate_counts),
        'tokenizer_replay': with_tokenizers,
        'scope': 'All NEW reconstruction records: inventory, complete condition grids, frozen input/log '
                 'bindings, lexical phi, record digests, EOS-inclusive candidate-score sums, exact '
                 'ranking/ties and response assembly. No comparisons with older reconstruction results. '
                 + ('All retained encoder inputs and candidate texts retokenized/decoded offline. ' if with_tokenizers else
                    'Encoder ID/string equivalence and candidate ID/text decoding not independently replayed in default mode. ')
                 + 'No model forward passes, new victim generation or semantic disclosure assessment.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--tokenizers', action='store_true', help='Also independently replay all encoder/decoder tokenization offline')
    args = parser.parse_args()
    try:
        report = check(args.root.resolve(), args.tokenizers)
    except (ValueError, KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
