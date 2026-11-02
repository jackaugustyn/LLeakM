"""Resumable offline T5 rerun from validated emitted-text length vectors.

This new campaign changes only the multi-candidate selector: confidence includes
the first EOS and excludes later forced PAD positions. References, length vectors,
seeds, model revisions and generation settings are frozen from the completed
2026-10-31 campaign. Every sampled candidate is retained for independent audit.
CPU and CUDA runs remain separate; partial results are never marked complete.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LEGACY = ROOT/'experiment_validation/revalidation_20261031'
from candidate_evidence import packed, record_digest, validate_record_candidates
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
WORD = re.compile(r"[A-Za-z0-9']+")


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cosine(a, b):
    x, y = Counter(WORD.findall(a.lower())), Counter(WORD.findall(b.lower()))
    denominator = math.sqrt(sum(v*v for v in x.values()) * sum(v*v for v in y.values()))
    return sum(v*y[k] for k, v in x.items()) / denominator if denominator else 0.0


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=LEGACY/'collection/canonical_traces.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='New records in this invocation; 0 means all pending.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--plan', action='store_true', help='Validate inventory and show counts without model loading.')
    args = parser.parse_args()
    source_runs = {
        (LEGACY/'collection/canonical_traces.jsonl').resolve(): 'cuda_full',
        (LEGACY/'fresh_references/tinyllama/canonical_traces.jsonl').resolve(): 'fresh_tinyllama_cuda',
        (LEGACY/'fresh_references/phi/canonical_traces.jsonl').resolve(): 'fresh_phi_cuda',
    }
    source_name = source_runs.get(args.input.resolve())
    with args.input.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    keys = [(r['label'], r['idx']) for r in rows]
    if len(set(keys)) != len(keys) or not rows:
        raise ValueError('Empty or duplicate input inventory')
    requested = {record_digest(row):(row['label'],row['idx']) for row in rows}
    original, origins, old_protocols, source_bindings = {}, {}, {}, {}
    for canonical, run_name in source_runs.items():
        if source_name is not None and source_name != run_name: continue
        old_run = LEGACY/'inference'/run_name
        old_protocol_path = old_run/'protocol.json'
        prior_protocol = json.loads(old_protocol_path.read_text())
        if sha_file(canonical) != prior_protocol['input_sha256']:
            raise ValueError('Frozen canonical input hash mismatch')
        matching = set()
        with canonical.open() as source:
            for line in source:
                if not line.strip(): continue
                frozen_row = json.loads(line)
                key = requested.get(record_digest(frozen_row))
                if key is not None:
                    if key in origins: raise ValueError('Ambiguous frozen input origin')
                    origins[key] = run_name; matching.add(key)
        if not matching: continue
        old_protocols[run_name] = prior_protocol
        for source_path in [canonical, old_protocol_path, old_run/'samples.jsonl']:
            source_bindings[str(source_path.relative_to(ROOT))] = sha_file(source_path)
        with (old_run/'samples.jsonl').open() as source:
            for line in source:
                if not line.strip(): continue
                record = json.loads(line); key = (record['label'], record['idx'])
                if key not in matching: continue
                if key in original: raise ValueError('Duplicate original selector record')
                original[key] = {'prediction':record['prediction'],'phi':record['phi'],
                                 'record_sha256':record_digest(record)}
    if set(keys) != set(original) or set(keys) != set(origins):
        raise ValueError('Every input must match an unchanged row of a frozen canonical inventory')
    old_protocol = next(iter(old_protocols.values()))
    for prior in old_protocols.values():
        for field in ['checkpoint_file_sha256','torch','transformers','threads']:
            if prior[field] != old_protocol[field]:
                raise ValueError('Mixed source protocols are not comparable: '+field)
    original_selector_path = HERE.parent/'before/weiss_reconstruction.py'
    for prior in old_protocols.values():
        if sha_file(original_selector_path) != prior['reconstructor_sha256']:
            raise ValueError('Original selector snapshot does not match source protocol')
    source_bindings[str(original_selector_path.relative_to(ROOT))] = sha_file(original_selector_path)
    for row in rows:
        if sha_bytes(row['reference'].encode()) != row['reference_sha256']:
            raise ValueError('Reference hash mismatch')
        if sha_bytes(json.dumps(row['token_lengths'], separators=(',', ':')).encode()) != row['token_lengths_sha256']:
            raise ValueError('Length-vector hash mismatch')
        if any(not isinstance(n, int) or n < 0 for n in row['token_lengths']):
            raise ValueError('Invalid length vector')
        if not isinstance(row.get('reconstruction_seed'), int):
            raise ValueError('Exact reconstruction seed missing')
    plan = {'records':len(rows), 'models':dict(Counter(r['label'] for r in rows)),
            'diagnostic_reference_records':sum(bool(r['diagnostic_series']) for r in rows),
            'device':args.device, 'all_segments':True, 'samples_per_segment':3,
            'encoder_and_decoder_max_length':80, 'input_sha256':sha_file(args.input),
            'scope':'EOS-bounded selection on frozen validated logged lengths and references; no new victim generation.',
            'selector_version':'eos-through-first-inclusive/1',
            'candidate_evidence_schema':'eos-bounded-candidate-audit/1',
            'diagnostic_subset':source_name is None,
            'original_selector_runs':sorted(old_protocols),
            'original_source_sha256':source_bindings}
    if args.plan:
        print(json.dumps(plan, indent=2)); return 0
    if args.limit < 0 or args.threads < 1:
        raise ValueError('Invalid execution limit or thread count')
    out = args.output.resolve()
    for protected in [ROOT/'experiment_validation/results', ROOT/'experiment_validation/analysis', LEGACY]:
        if out == protected or protected in out.parents:
            raise ValueError('Use a separate output directory, not archived results/analysis')
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    lock_handle = (out/'.run.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from huggingface_hub import hf_hub_download
    import weiss_reconstruction as wr
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this process; no automatic CPU fallback or mixed-device resume.')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ['LLEAKM_RECONSTRUCT_DEVICE'] = args.device
    checkpoint_revisions = {
        wr.FIRST_SENTENCES_MODEL:'057cf090b719f5cf89da504babd2122d404e0953',
        wr.MIDDLE_SENTENCES_MODEL:'07e2b3fc96bc129eb4a2871e58731c07e889cecc',
    }
    checkpoint_files = {
        model_id: {name: sha_file(hf_hub_download(model_id, name, revision=revision, local_files_only=True))
                   for name in ['pytorch_model.bin', 'tokenizer.json', 'config.json']}
        for model_id, revision in checkpoint_revisions.items()
    }
    if checkpoint_files != old_protocol['checkpoint_file_sha256']:
        raise ValueError('Checkpoint file hashes differ from the frozen original protocol')
    if torch.__version__ != old_protocol['torch'] or transformers.__version__ != old_protocol['transformers']:
        raise ValueError('Software differs from the original campaign; no mixed-environment comparison')
    if args.threads != old_protocol['threads']:
        raise ValueError('Thread count must match the frozen original protocol')
    protocol = dict(plan, torch=torch.__version__, transformers=transformers.__version__,
                    threads=args.threads, checkpoints=checkpoint_revisions,
                    checkpoint_file_sha256=checkpoint_files,
                    reconstructor_sha256=sha_file(ROOT/'weiss_reconstruction.py'),
                    runner_sha256=sha_file(__file__), evidence_validator_sha256=sha_file(HERE/'candidate_evidence.py'), precision='float32',
                    seed_source='recorded per-sample reconstruction_seed')
    protocol_id = sha_bytes(json.dumps(protocol, sort_keys=True).encode())
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Output protocol differs: choose a new directory; old results were not modified.')
    write_json(manifest, protocol)
    samples = out/'samples.jsonl'
    completed = {}
    by_key = {(row['label'], row['idx']): row for row in rows}
    if samples.exists():
        for line in samples.open():
            record = json.loads(line)
            key = (record['label'],record['idx'])
            if record['protocol_id'] != protocol_id or key not in keys or key in completed:
                raise ValueError('Invalid or duplicate resumed record')
            source = by_key[key]
            for field in ['reference_sha256', 'generation_log_sha256', 'reference', 'token_lengths', 'reconstruction_seed']:
                if record[field] != source[field]:
                    raise ValueError(f'Resumed record does not match its source: {key}, {field}')
            if record['phi'] != cosine(record['reference'], record['prediction']):
                raise ValueError(f'Resumed metric does not match saved texts: {key}')
            if record.get('record_sha256') != record_digest({k:v for k,v in record.items() if k!='record_sha256'}):
                raise ValueError('Resumed record digest mismatch')
            prior = original[key]
            if record.get('original_selector_run') != origins[key]:
                raise ValueError('Original comparison run mismatch')
            for old_field, new_field in [('prediction','original_selector_prediction'),('phi','original_selector_phi'),('record_sha256','original_selector_record_sha256')]:
                if record.get(new_field) != prior[old_field]: raise ValueError('Original comparison record mismatch')
            validate_record_candidates(record)
            completed[key] = {'label':record['label'],'phi':record['phi']}
    status = dict(protocol_id=protocol_id, expected=len(rows), completed=len(completed),
                  complete=len(completed)==len(rows), status='starting',
                  scope=plan['scope'], new_records_this_invocation=0,pid=os.getpid())
    write_json(out/'status.json', status)

    def load_bundle(model_id):
        if model_id not in wr._MODEL_CACHE:
            revision = checkpoint_revisions[model_id]
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, use_fast=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision,
                local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float32)
            model.to(args.device).eval()
            wr._MODEL_CACHE[model_id] = (model,tokenizer,args.device)
        return wr._MODEL_CACHE[model_id]

    wr._load_model_bundle = load_bundle
    actual_inputs = []
    original_sample = wr._sample_and_rank

    def observed_sample(model, tokenizer, device, input_text, *, samples, max_length=80):
        full = tokenizer([input_text], padding=True, truncation=False)['input_ids'][0]
        kept = tokenizer([input_text], max_length=max_length, padding=True, truncation=True)['input_ids'][0]
        entry = dict(text=input_text,full_token_ids=full,full_token_count=len(full),retained_token_ids=kept,
                     truncated=len(full)>len(kept),max_length=max_length,samples=samples,
                     model_id=next(k for k,v in wr._MODEL_CACHE.items() if v[0] is model))
        actual_inputs.append(entry)
        candidate_audit = []
        ranked = original_sample(model,tokenizer,device,input_text,samples=samples,max_length=max_length,
                                 candidate_audit=candidate_audit)
        if len(candidate_audit) != 1: raise ValueError('One audit object required per decoder call')
        entry['candidate_audit'] = candidate_audit[0]
        entry['selected_text'] = ranked[0][0] if ranked else ''
        return ranked

    wr._sample_and_rank = observed_sample
    start = time.monotonic()
    try:
        if len(completed) < len(rows):
            status['status'] = 'loading_pinned_models'
            write_json(out/'status.json', status)
            # Keep loading outside per-record seeded reconstruction. This is
            # explicit even though reconstruct() also seeds after cache loads.
            for model_id in checkpoint_revisions:
                load_bundle(model_id)
        for row in rows:
            key=(row['label'],row['idx'])
            if key in completed:
                continue
            if args.limit and status['new_records_this_invocation'] >= args.limit:
                break
            status.update(status='running', active={'label':row['label'],'idx':row['idx']})
            write_json(out/'status.json',status)
            actual_inputs.clear()
            t = time.monotonic()
            result = wr.reconstruct(row['token_lengths'],num_first_candidates=3,max_sentences=0,
                                    samples_per_segment=3,seed=row['reconstruction_seed'])
            rec = dict(label=row['label'],model=row['model'],idx=row['idx'],topic=row['topic'],
                run_id=row['run_id'],protocol_id=protocol_id,reconstruction_seed=row['reconstruction_seed'],
                generation_log_sha256=row['generation_log_sha256'],reference_sha256=row['reference_sha256'],
                diagnostic_series=row['diagnostic_series'],reference_validation=row['reference_validation'],
                reference=row['reference'],token_lengths=row['token_lengths'],
                prediction=result.full_text,first_prediction=result.first_sentence,
                reconstructed_segments=result.sentence_count,available_segments=result.available_segment_count,
                phi=cosine(row['reference'],result.full_text),
                old_phi=cosine(row['reference'],row['old_pred_full_text']),
                original_selector_run=origins[key],
                original_selector_prediction=original[key]['prediction'],
                original_selector_phi=original[key]['phi'],
                original_selector_record_sha256=original[key]['record_sha256'],
                actual_encoder_inputs=list(actual_inputs),elapsed_seconds=time.monotonic()-t)
            validate_record_candidates(rec)
            if rec['prediction'] != wr._concat_segments([entry['selected_text'] for entry in actual_inputs]):
                raise ValueError('Assembled prediction differs from selected segments')
            rec['record_sha256'] = record_digest(rec)
            with samples.open('a') as handle:
                handle.write(packed(rec)+'\n');handle.flush();os.fsync(handle.fileno())
            completed[key] = {'label':rec['label'],'phi':rec['phi']}
            status['new_records_this_invocation'] += 1
            status.update(completed=len(completed),complete=len(completed)==len(rows),
                          elapsed_seconds=time.monotonic()-start)
            write_json(out/'status.json',status)
            print(f"{len(completed)}/{len(rows)} {key} phi={rec['phi']:.6f} seconds={rec['elapsed_seconds']:.1f}",flush=True)
        status['status'] = 'completed' if status['complete'] else 'partial'
        status.pop('active',None)
        write_json(out/'status.json',status)
    except BaseException as exc:
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-start)
        write_json(out/'status.json',status)
        raise
    write_json(out/'summary.json',dict(complete=status['complete'],n=len(completed),expected=len(rows),
        diagnostic_series_included=True,scope=plan['scope'],
        models={label:{'n':len(rr),'mean_phi':sum(r['phi'] for r in rr)/len(rr)}
                for label in sorted({r['label'] for r in completed.values()})
                if (rr:=[r for r in completed.values() if r['label']==label])}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
