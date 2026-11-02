"""Offline, bounded first-segment feasibility check; never a replacement table."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['LLEAKM_RECONSTRUCT_DEVICE'] = 'cpu'
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import weiss_reconstruction as wr

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
started = time.monotonic()
row = json.loads((ROOT / 'experiment_validation/results/run_full_20260920_qwen_1_5b_full/samples.jsonl').read_text().splitlines()[0])
log_path = ROOT / 'logs' / (row['run_id'] + '.json')
raw = log_path.read_bytes()
log = json.loads(raw)
lengths = [s['token_utf8_len'] for s in log['steps']]
segments = wr.heuristic_sentences_from_lengths(lengths)
model_path = Path('/home/paugustynowicz/.cache/huggingface/hub/models--royweiss1--T5_FirstSentences/snapshots/057cf090b719f5cf89da504babd2122d404e0953')
status = dict(status='loading', started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              mode='CPU float32, four threads, offline', scope='one first segment, three candidates; not a campaign rerun',
              model_path=str(model_path), sample_run_id=row['run_id'], log_sha256=hashlib.sha256(raw).hexdigest(),
              token_lengths=lengths, segment=segments[0], available_segments=len(segments), seed=20260706)


def save():
    status['elapsed_seconds'] = time.monotonic() - started
    (OUT / 'cpu_probe.json').write_text(json.dumps(status, indent=2) + '\n')
    print(status['status'], round(status['elapsed_seconds'], 2), flush=True)


save()
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True, low_cpu_mem_usage=True)
    model.eval()
    status['status'] = 'inference'
    status['load_seconds'] = time.monotonic() - started
    save()
    wr.seed_reconstructor(status['seed'])
    generated = wr._sample_and_rank(model, tokenizer, 'cpu', wr.make_input_from_lengths(segments[0]), samples=3)
    status['status'] = 'completed'
    status['candidates'] = generated
    status['inference_seconds'] = time.monotonic() - started - status['load_seconds']
except Exception as exc:
    status['status'] = 'failed'
    status['exception_type'] = type(exc).__name__
    status['exception'] = str(exc)
    save()
    raise
save()
