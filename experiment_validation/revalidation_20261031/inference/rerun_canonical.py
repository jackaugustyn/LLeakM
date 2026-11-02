"""Resumable offline T5 rerun from validated emitted-text length vectors.

This repairs the observation source only. It cannot reconstruct missing original
token IDs or certify ordinary reference decoding. CPU and CUDA runs are kept
separate, and partial results never become a complete campaign.
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
    parser.add_argument('--input', type=Path, default=HERE.parent/'collection/canonical_traces.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='New records in this invocation; 0 means all pending.')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--plan', action='store_true', help='Validate inventory and show counts without model loading.')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    keys = [(r['label'], r['idx']) for r in rows]
    if len(set(keys)) != len(keys) or not rows:
        raise ValueError('Empty or duplicate input inventory')
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
            'scope':'Verified logged emitted-text lengths; ordinary reference decoding remains unverified.'}
    if args.plan:
        print(json.dumps(plan, indent=2)); return 0
    if args.limit < 0 or args.threads < 1:
        raise ValueError('Invalid execution limit or thread count')
    out = args.output.resolve()
    for protected in [ROOT/'experiment_validation/results', ROOT/'experiment_validation/analysis']:
        if out == protected or protected in out.parents:
            raise ValueError('Use a separate output directory, not archived results/analysis')
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
    protocol = dict(plan, torch=torch.__version__, transformers=transformers.__version__,
                    threads=args.threads, checkpoints=checkpoint_revisions,
                    checkpoint_file_sha256=checkpoint_files,
                    reconstructor_sha256=sha_file(ROOT/'weiss_reconstruction.py'),
                    runner_sha256=sha_file(__file__), precision='float32',
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
        for line in samples.read_text().splitlines():
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
            completed[key] = record
    status = dict(protocol_id=protocol_id, expected=len(rows), completed=len(completed),
                  complete=len(completed)==len(rows), status='starting',
                  scope=plan['scope'], new_records_this_invocation=0)
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
        actual_inputs.append(dict(text=input_text, full_token_count=len(full), retained_token_ids=kept,
                                  truncated=len(full)>len(kept), max_length=max_length))
        return original_sample(model,tokenizer,device,input_text,samples=samples,max_length=max_length)

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
                actual_encoder_inputs=list(actual_inputs),elapsed_seconds=time.monotonic()-t)
            with samples.open('a') as handle:
                handle.write(json.dumps(rec,ensure_ascii=False)+'\n');handle.flush();os.fsync(handle.fileno())
            completed[key] = rec
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
