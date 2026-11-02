"""Independently validate canonical T5 reruns without loading model weights.

Default success requires all inventory records and the runner's final completion
status. --allow-partial permits only a clearly marked diagnostic snapshot.
Archived emitted references remain the estimand, including two diagnostic
series. This verifier does not certify the missing historical token IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(ROOT))
from experiment_validation.scripts.generate_prompts import TOPICS
from experiment_validation.scripts import run_validation as original_metrics
from experiment_validation.scripts.analyze_stats import RUNS

spec = importlib.util.spec_from_file_location(
    'independent_statistics', BASE / 'statistics/recompute_all_statistics.py')
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)

KNOWN_MODELS = {label: 300 for label, _, _ in RUNS}
FIRST = 'royweiss1/T5_FirstSentences'
MIDDLE = 'royweiss1/T5_MiddleSentences'
PREFIX_FIRST = 'Translate the Special Tokens to English. \nSpecial Tokens:'
PREFIX_MIDDLE = 'Translate the Special Tokens to English, given the context. \nContext: '


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_segments(lengths):
    """Restate the published length-1 split and forward short-segment merge."""
    cuts = [-1] + [i for i, value in enumerate(lengths) if value == 1]
    if cuts[-1] != len(lengths) - 1:
        cuts.append(len(lengths) - 1)
    pieces = [lengths[a + 1:b + 1] for a, b in zip(cuts, cuts[1:])]
    for i in range(len(pieces) - 1):
        if pieces[i][-3:] == [3, 1, 1]:
            pieces[i + 1] = pieces[i][-3:] + pieces[i + 1]
            pieces[i] = pieces[i][:-3]
    pending, result = [], []
    for piece in filter(None, pieces):
        pending += piece
        if len(pending) >= 10:
            result.append(pending)
            pending = []
    if pending:
        result.append(pending)
    return result


def first_sentence(text):
    text = text.strip()
    if not text:
        return ''
    match = re.search(r'(.+?[.!?])(?:\s|$)', text, flags=re.S)
    return match.group(1).strip() if match else text.splitlines()[0].strip()


def normalize_parts(parts):
    return re.sub(r'\s+([,.;:!?])', r'\1', ' '.join(s.strip() for s in parts if s.strip()))


def scenario(source):
    matches = [x for x in TOPICS[source['topic']]['issues'] if x.lower() in source['prompt'].lower()]
    require(len(matches) == 1, f"Ambiguous scenario: {source['label']}/{source['idx']}")
    return source['topic'] + '::' + matches[0]


def checkpoint_tokenizers(protocol):
    cache = Path(os.environ.get('HF_HUB_CACHE', str(Path.home() / '.cache/huggingface/hub')))
    result = {}
    checked = {}
    for model, revision in protocol['checkpoints'].items():
        folder = cache / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
        for filename, expected in protocol['checkpoint_file_sha256'][model].items():
            path = folder / filename
            require(file_digest(path) == expected, f'Checkpoint hash mismatch: {path}')
            checked[str(path)] = expected
        raw = Tokenizer.from_file(str(folder / 'tokenizer.json'))
        raw.no_padding()
        raw.no_truncation()
        limited = Tokenizer.from_file(str(folder / 'tokenizer.json'))
        limited.no_padding()
        limited.enable_truncation(max_length=80, direction='right')
        result[model] = (raw, limited)
    return result, checked


def validate_fresh_reference(row, log, cached):
    """Replay pinned victim-tokenizer decoding without loading victim weights."""
    require(row.get('source_kind') == 'fresh_contextual_reference', 'Fresh source kind missing')
    require(row.get('historical_comparison_available') is False, 'Fresh references cannot be historical paired predictions')
    require(row['reference_validation'].startswith('contextual-full-sequence-decoding-verified-v1'), 'Fresh reference contract missing')
    run_dir = Path(row['run_dir'])
    if run_dir not in cached:
        protocol = json.loads((run_dir/'protocol.json').read_text())
        require(digest(json.dumps(protocol,sort_keys=True).encode()) == row['protocol_id'], 'Fresh generation protocol ID mismatch')
        require(file_digest(ROOT/'app.py') == protocol['harness_source_sha256'], 'Fresh harness source changed')
        require(file_digest(BASE/'fresh_references/generate_fresh.py') == protocol['runner_sha256'], 'Fresh generation runner changed')
        cache = Path(os.environ.get('HF_HUB_CACHE',str(Path.home()/'.cache/huggingface/hub')))
        folder = cache/('models--'+protocol['model_id'].replace('/','--'))/'snapshots'/protocol['revision']
        for filename, expected in protocol['checkpoint_file_sha256'].items():
            require(file_digest(folder/filename)==expected, f'Fresh victim checkpoint hash: {filename}')
        tokenizer = Tokenizer.from_file(str(folder/'tokenizer.json'))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        cached[run_dir] = protocol, tokenizer
    protocol, tokenizer = cached[run_dir]
    require(digest(json.dumps(protocol,sort_keys=True).encode()) == row['protocol_id'], 'Mixed generation protocols')
    require(row['model_id']==protocol['model_id']==log['model_id'], 'Fresh model identity mismatch')
    require(log['harness_source_sha256']==protocol['harness_source_sha256'], 'Fresh log harness identity mismatch')
    require(log['detokenization_policy']=='contextual_prefix_utf8_buffer_v1', 'Fresh log detokenization contract')
    ids = row['generated_token_ids']
    require(ids==log['generated_token_ids']==[s['token_id'] for s in log['steps']], 'Fresh token-ID mismatch')
    require(len(ids)==log['generated_tokens']==len(row['token_lengths']), 'Fresh generated-token count')
    require(tokenizer.decode(ids,skip_special_tokens=False)==row['reference'], 'Fresh full-sequence tokenizer decoding')
    require(tokenizer.encode(row['input_text']).ids==row['input_token_ids'], 'Fresh input tokenization')
    frames = []
    for i, step in enumerate(log['steps']):
        frame = b'data: '+step['token_text'].encode('utf-8')+b'\n\n'
        require(step['step']==i and step['sse_frame_utf8_len']==len(frame), 'Fresh step/frame accounting')
        frames.append(frame)
    frames.append(b'data: [DONE]\n\n')
    wire = b''.join(frames)
    require(len(wire)==row['emitted_stream_bytes'] and digest(wire)==row['emitted_stream_sha256'], 'Fresh emitted stream hash')


def check_inventory(inventory, protocol, *, historical_main):
    keys = [(r['label'], r['idx']) for r in inventory]
    expected_models = dict(Counter(r['label'] for r in inventory))
    require(keys and len(keys) == len(set(keys)), 'Inventory must contain unique nonempty keys')
    require(set(expected_models) <= set(KNOWN_MODELS) and set(expected_models.values()) == {300}, 'Expected 300 records for each known model in inventory')
    if historical_main:
        require(expected_models == KNOWN_MODELS, 'Historical main inventory requires all seven models')
    require(protocol['records'] == len(keys) and protocol['models'] == expected_models, 'Wrong protocol inventory')
    require(protocol['all_segments'] is True and protocol['samples_per_segment'] == 3, 'Wrong segment/candidate protocol')
    require(protocol['encoder_and_decoder_max_length'] == 80 and protocol['precision'] == 'float32', 'Wrong model length/precision protocol')
    require(file_digest(ROOT / 'weiss_reconstruction.py') == protocol['reconstructor_sha256'], 'Reconstructor source changed')
    require(file_digest(BASE / 'inference/rerun_canonical.py') == protocol['runner_sha256'], 'Runner source changed')
    cached_archives = {}
    cached_fresh = {}
    by_key = {}
    prompts_by_idx = {}
    for row in inventory:
        key = row['label'], row['idx']
        require(digest(row['reference'].encode()) == row['reference_sha256'], f'Reference hash: {key}')
        require(digest(json.dumps(row['token_lengths'], separators=(',', ':')).encode()) == row['token_lengths_sha256'], f'Vector hash: {key}')
        require(all(type(x) is int and x >= 0 for x in row['token_lengths']), f'Invalid vector: {key}')
        log_path = ROOT / row['generation_log_path']
        log_raw = log_path.read_bytes()
        require(digest(log_raw) == row['generation_log_sha256'], f'Generation log hash: {key}')
        log = json.loads(log_raw)
        emitted = [step['token_text'] for step in log['steps']]
        require(''.join(emitted) == row['reference'] == log['response_text'], f'Log reference: {key}')
        require([len(s.encode('utf-8')) for s in emitted] == row['token_lengths'], f'Log lengths: {key}')
        require([s['token_utf8_len'] for s in log['steps']] == row['token_lengths'], f'Logged payload accounting: {key}')
        require(log['run_id'] == row['run_id'] and log['prompt'] == row['prompt'], f'Log identity: {key}')
        require(log['finish_reason']==row['finish_reason'] and log['response_complete']==row['response_complete'], f'Log termination: {key}')
        row['_has_old_comparison'] = bool(row.get('old_pred_full_text'))
        if historical_main or row.get('sample_file_sha256'):
            path = ROOT / 'experiment_validation/results' / row['run_dir'] / 'samples.jsonl'
            if path not in cached_archives:
                archive_raw = path.read_bytes()
                require(digest(archive_raw) == row['sample_file_sha256'], f'Archive hash: {path}')
                cached_archives[path] = {r['idx']: r for r in map(json.loads, archive_raw.splitlines())}
            old = cached_archives[path][row['idx']]
            require(old['response_text'] == row['reference'] and old['pred_full_text'] == row['old_pred_full_text'], f'Archive texts: {key}')
            require(first_sentence(row['reference']) == old['response_first_sentence'], f'Archived first sentence: {key}')
        else:
            require(not row['_has_old_comparison'], f'Fresh inventory cannot claim an unbound historical comparison: {key}')
            validate_fresh_reference(row,log,cached_fresh)
            path = ROOT/'experiment_validation/results'/row['historical_run_dir']/'samples.jsonl'
            if path not in cached_archives:
                archive_raw = path.read_bytes()
                require(digest(archive_raw)==row['historical_sample_file_sha256'], f'Fresh historical archive hash: {key}')
                cached_archives[path] = {r['idx']:r for r in map(json.loads,archive_raw.splitlines())}
            old = cached_archives[path][row['idx']]
            require(old['run_id']==row['historical_run_id'], f'Fresh historical run binding: {key}')
            require(digest(old['response_text'].encode())==row['historical_reference_sha256'], f'Fresh historical reference binding: {key}')
            for name in ['idx','topic','prompt','reconstruction_seed']:
                require(old[name]==row[name], f'Fresh historical settings binding: {key}/{name}')
        row['_scenario'] = scenario(row)
        prompt_id = row['topic'], row['prompt'], row['_scenario']
        require(prompts_by_idx.setdefault(row['idx'], prompt_id) == prompt_id, f'Unpaired model prompt: {key}')
        by_key[key] = row
    for label in expected_models:
        rows = [r for r in inventory if r['label'] == label]
        require({r['idx'] for r in rows} == set(range(1, 301)), f'Unexpected indices: {label}')
        counts = Counter(r['_scenario'] for r in rows)
        require(len(counts) == 60 and set(counts.values()) == {5}, f'Scenario balance: {label}')
    return by_key


def validate_record(row, source, protocol_id, tokenizers):
    key = row['label'], row['idx']
    require(row['protocol_id'] == protocol_id, f'Protocol ID mismatch: {key}')
    for name in ['label', 'idx', 'model', 'topic', 'run_id', 'reference', 'reference_sha256',
                 'generation_log_sha256', 'token_lengths', 'reconstruction_seed',
                 'diagnostic_series', 'reference_validation']:
        require(row[name] == source[name], f'Source mismatch: {key}/{name}')
    metric = independent.metric(source['reference'], row['prediction'])
    old_metric = independent.metric(source['reference'], source.get('old_pred_full_text',''))
    first = independent.metric(first_sentence(source['reference']), row['first_prediction'])
    require(math.isclose(metric['phi'], row['phi'], abs_tol=1e-12, rel_tol=0), f'Phi mismatch: {key}')
    require(math.isclose(old_metric['phi'], row['old_phi'], abs_tol=1e-12, rel_tol=0), f'Old phi mismatch: {key}')
    segments = independent_segments(source['token_lengths'])
    calls = row['actual_encoder_inputs']
    available, reconstructed = row['available_segments'], row['reconstructed_segments']
    require(available == len(segments), f'Available segment count: {key}')
    require(0 <= reconstructed <= len(calls) <= available, f'Impossible segment/call count: {key}')
    if available:
        require(reconstructed >= 1 and len(calls) >= 1, f'Missing first segment call: {key}')
        # An empty middle decode is attempted but not appended to the result.
        require(len(calls) in {reconstructed, reconstructed + 1}, f'Unexpected failed calls: {key}')
        if len(calls) < available:
            require(len(calls) == reconstructed + 1, f'Unexplained early stop: {key}')
    else:
        require(not row['prediction'] and not row['first_prediction'], f'Nonempty output from empty input: {key}')
    contexts = []
    total_full, total_kept, truncations = 0, 0, 0
    for i, (call, lengths) in enumerate(zip(calls, segments)):
        tokens = ''.join(' _' + str(n) for n in lengths)
        require(call['max_length'] == 80, f'Encoder cap: {key}/{i}')
        if i == 0:
            require(call['text'] == PREFIX_FIRST + tokens, f'First input serialization: {key}')
            tokenizer = tokenizers[FIRST]
        else:
            suffix = ' \nSpecial Tokens:' + tokens
            require(call['text'].startswith(PREFIX_MIDDLE) and call['text'].endswith(suffix), f'Middle input serialization: {key}/{i}')
            context = call['text'][len(PREFIX_MIDDLE):-len(suffix)]
            contexts.append(context)
            if i == 1:
                require(context == row['first_prediction'], f'First context mismatch: {key}')
            tokenizer = tokenizers[MIDDLE]
        full = tokenizer[0].encode(call['text']).ids
        kept = tokenizer[1].encode(call['text']).ids
        require(len(full) == call['full_token_count'], f'Full tokenizer length: {key}/{i}')
        require(kept == call['retained_token_ids'], f'Encoder token IDs: {key}/{i}')
        require((len(full) > len(kept)) == call['truncated'], f'Truncation indicator: {key}/{i}')
        total_full += len(full)
        total_kept += len(kept)
        truncations += bool(call['truncated'])
    if contexts:
        require(row['prediction'].startswith(normalize_parts(contexts)), f'Generated context chain mismatch: {key}')
        if len(calls)>reconstructed:
            require(row['prediction']==normalize_parts(contexts), f'Empty-candidate stop output mismatch: {key}')
    if reconstructed == 1:
        require(row['prediction'] == normalize_parts([row['first_prediction']]), f'Single segment concatenation mismatch: {key}')
    return dict(label=row['label'], model=row['model'], idx=row['idx'], topic=row['topic'],
                scenario=source['_scenario'], run_id=row['run_id'], diagnostic_series=row['diagnostic_series'],
                finish_reason=source['finish_reason'], response_complete=source['response_complete'],
                token_count=len(source['token_lengths']), metrics=metric, first_metrics=first,
                old_metrics=old_metric if source['_has_old_comparison'] else None,
                delta_phi=metric['phi']-old_metric['phi'] if source['_has_old_comparison'] else None,
                available_segments=available, reconstructed_segments=reconstructed,
                attempted_segments=len(calls), unattempted_segments=available-len(calls),
                early_stop=len(calls)<available, empty_candidate_stop=len(calls)>reconstructed,
                truncated_encoder_calls=truncations, encoder_tokens_full=total_full,
                encoder_tokens_retained=total_kept)


def aggregate(scored):
    """All-record summaries; scenario resampling keeps the five variants together."""
    rng = np.random.default_rng(independent.SEED)
    models, grouped, arrays = [], {}, {}
    labels = [label for label in KNOWN_MODELS if any(r['label']==label for r in scored)]
    for label in labels:
        rows = sorted([r for r in scored if r['label'] == label], key=lambda r:r['idx'])
        grouped[label] = np.array([np.mean([r['metrics']['phi'] for r in rows if r['scenario'] == s])
                                   for s in sorted({r['scenario'] for r in rows})])
        arrays[label] = np.array([r['metrics']['phi'] for r in rows])
        model = dict(label=label, model=rows[0]['model'], n=300, n_scenarios=60,
                     diagnostic_series=rows[0]['diagnostic_series'], metrics={}, first_metrics={},
                     phi_ci_scenario_bootstrap=independent.ci(grouped[label], rng),
                     old_phi_mean=float(np.mean([r['old_metrics']['phi'] for r in rows])) if all(r['old_metrics'] is not None for r in rows) else None,
                     delta_phi_mean=float(np.mean([r['delta_phi'] for r in rows])) if all(r['delta_phi'] is not None for r in rows) else None,
                     LR_at_0_5=float(np.mean(arrays[label]>=.5)),
                     eos_count=sum(r['response_complete'] for r in rows),
                     early_stop_count=sum(r['early_stop'] for r in rows),
                     empty_candidate_stop_count=sum(r['empty_candidate_stop'] for r in rows))
        for scope in ['metrics', 'first_metrics']:
            for metric in rows[0][scope]:
                values = [r[scope][metric] for r in rows]
                model[scope][metric] = {'mean':float(np.mean(values)),
                                        'ci_prompt_bootstrap':independent.ci(values, rng)}
        eos = [r['metrics']['phi'] for r in rows if r['response_complete']]
        model['eos_only_phi_mean'] = float(np.mean(eos)) if eos else None
        model['topics'] = {t:{'n':len(values), 'phi_mean':float(np.mean(values))}
                          for t in sorted({r['topic'] for r in rows})
                          if (values := [r['metrics']['phi'] for r in rows if r['topic']==t])}
        models.append(model)
    pairs, scenario_pairs = [], []
    for a, b in itertools.combinations(labels, 2):
        for table, values in [(pairs, arrays), (scenario_pairs, grouped)]:
            diff = values[a]-values[b]
            table.append(dict(a=a,b=b,mean_diff_phi=float(diff.mean()),
                              diff_ci_bootstrap=independent.ci(diff,rng),
                              **independent.signed_rank(values[a],values[b])))
    independent.holm(pairs)
    independent.holm(scenario_pairs)
    return {'method':f'Descriptive conditional rerun, 10000 percentile bootstrap draws; paired signed ranks with tie-adjusted normal approximation and continuity correction; Holm over {len(pairs)} pairs. Scenario sensitivity averages five variants within each of 60 scenarios before resampling/testing.',
            'bootstrap_seed':independent.SEED, 'bootstrap_draws':independent.BOOT,
            'models':models, 'pairwise_prompt_tests':pairs, 'pairwise_scenario_tests':scenario_pairs,
            'limitations':['Models marked diagnostic retain malformed historical references; differences do not establish comparative model vulnerability.',
                           'Changed collector inputs, environment and random state are not causally separated.',
                           'Encoder truncation at 80 tokens remains; full segment request is not full source coverage.',
                           'Prompt/bootstrap intervals and scenario/bootstrap intervals are different exploratory uncertainty estimates.',
                           'LR at 0.5 is a lexical threshold frequency, not calibrated sensitive-data disclosure.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=BASE/'inference/cuda_full')
    p.add_argument('--inventory', '--input', dest='inventory', type=Path, default=BASE/'collection/canonical_traces.jsonl')
    p.add_argument('--output', type=Path, default=HERE/'cuda_full')
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    require(out != args.run.resolve() and args.run.resolve() not in out.parents, 'Verifier output must be outside inference source directory')
    for protected in [ROOT/'experiment_validation/results', ROOT/'experiment_validation/analysis']:
        require(out != protected and protected not in out.parents, 'Never overwrite archived results or analyses')
    out.mkdir(parents=True, exist_ok=True)
    report = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),
              'verification_status':'FAILED', 'complete':False,
              'scope':'Canonical rerun bound to supplied emitted-text references; interpretation follows source reference_validation.',
              'run':str(args.run.resolve()), 'verifier_sha256':file_digest(__file__)}
    try:
        protocol = json.loads((args.run/'protocol.json').read_text())
        protocol_id = digest(json.dumps(protocol,sort_keys=True).encode())
        report['protocol_id'] = protocol_id
        if (out/'verification.json').exists():
            previous = json.loads((out/'verification.json').read_text())
            require(previous.get('protocol_id') in {None, protocol_id}, 'Use a separate output for a different protocol')
        raw = args.inventory.read_bytes()
        require(digest(raw) == protocol['input_sha256'], 'Inventory hash does not match protocol')
        inventory = [json.loads(x) for x in raw.splitlines() if x.strip()]
        expected = len(inventory)
        report['expected'] = expected
        historical_main = args.inventory.resolve() == (BASE/'collection/canonical_traces.jsonl').resolve()
        by_key = check_inventory(inventory,protocol,historical_main=historical_main)
        tokenizers, checked_files = checkpoint_tokenizers(protocol)
        independent.metric_selfcheck(original_metrics)
        source = args.run/'samples.jsonl'
        raw_samples = source.read_bytes() if source.exists() else b''
        # An active writer can be between write() and flush(). Never accept a
        # truncated final record as complete or mutate the source to repair it.
        tail = b''
        if raw_samples and not raw_samples.endswith(b'\n'):
            split = raw_samples.rfind(b'\n')
            raw_samples, tail = raw_samples[:split+1], raw_samples[split+1:]
        records = [json.loads(x) for x in raw_samples.splitlines() if x.strip()]
        scored, seen = [], set()
        for row in records:
            key = row['label'], row['idx']
            require(key in by_key and key not in seen, f'Unexpected/duplicate key: {key}')
            seen.add(key)
            scored.append(validate_record(row,by_key[key],protocol_id,tokenizers))
        status = json.loads((args.run/'status.json').read_text())
        require(status['protocol_id'] == protocol_id and status['expected'] == expected, 'Status protocol mismatch')
        complete = len(scored)==expected and not tail and status['complete'] is True and status['status']=='completed'
        if status['status'] == 'completed':
            require(complete, 'Runner claims completion with incomplete records')
        report.update(verification_status='VERIFIED_COMPLETE' if complete else 'PARTIAL_DIAGNOSTIC',
                      integrity_checks_passed=True, complete=complete, completed=len(scored),
                      model_counts=dict(Counter(r['label'] for r in scored)),
                      runner_status=status, ignored_incomplete_tail_bytes=len(tail),
                      sampled_jsonl_prefix_sha256=digest(raw_samples), checkpoint_files=checked_files,
                      metric_crosschecks=605, all_records_requested=True,
                      full_segment_coverage_count=sum(r['attempted_segments']==r['available_segments'] for r in scored),
                      early_stop_count=sum(r['early_stop'] for r in scored),
                      empty_candidate_stop_count=sum(r['empty_candidate_stop'] for r in scored),
                      attempted_segments=sum(r['attempted_segments'] for r in scored),
                      available_segments=sum(r['available_segments'] for r in scored),
                      truncated_encoder_calls=sum(r['truncated_encoder_calls'] for r in scored),
                      diagnostic_reference_records=sum(r['diagnostic_series'] for r in scored))
        target = out/'rescored.jsonl'
        tmp = target.with_suffix('.jsonl.tmp')
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in scored))
        tmp.replace(target)
        if complete:
            summary = aggregate(scored)
            save(out/'statistics.json',dict(complete=True,protocol_id=protocol_id,**summary))
        save(out/'verification.json',report)
        text = ['## Material Passport', '',
                '- Origin Skill: experiment-agent', '- Origin Mode: validate',
                f"- Origin Date: {report['verified_at_utc']}",
                '- Verification Status: '+('VERIFIED' if complete else 'ANALYZED'),
                '- Version Label: canonical_inference_validation_v1', '',
                '## Weryfikacja nowych rekonstrukcji T5', '',
                f"Stan: **{report['verification_status']}**, zapisane rekordy **{len(scored)}/{expected}**.", '',
                'Sprawdzono hashe źródeł i checkpointów, powiązanie wszystkich wejść z logami, unikalność rekordów, segmentację, rzeczywiste serializacje i identyfikatory enkodera oraz metryki obliczone niezależnie z tekstów.', '',
                f"Wywołania enkodera: {report['attempted_segments']}; obcięte do 80 tokenów: {report['truncated_encoder_calls']}; odpowiedzi z nieprzetworzonymi końcowymi segmentami: {report['early_stop_count']}.", '',
                'Historyczne referencje TinyLlama i Phi-3.5 mają status diagnostyczny; nowych generacji nie utożsamia się z naprawionymi historycznymi odpowiedziami. Nie przypisuje się całej zmiany wyniku wyłącznie naprawie kolektora.', '',
                'Pełne agregaty zapisano w statistics.json.' if complete else 'Wynik jest częściowy; nie wygenerowano agregatów pełnej kampanii.', '']
        (out/'WERYFIKACJA.md').write_text('\n'.join(text))
        print(json.dumps({k:report[k] for k in ['verification_status','complete','completed','expected','early_stop_count','truncated_encoder_calls']},indent=2))
        return 0 if complete or args.allow_partial else 2
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc))
        save(out/'verification.json',report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
