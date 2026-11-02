"""Rebuild and verify the outcome-independent, 24-case input inventory."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = ROOT/'experiment_validation/revalidation_20261031'


def build_inventory():
    source_names = ['inference/cuda_full', 'inference/fresh_tinyllama_cuda',
                    'inference/fresh_phi_cuda', 'secondary/cuda_full']
    sources = {name:[json.loads(x) for x in (BASE/name/'samples.jsonl').read_text().splitlines()]
               for name in source_names}
    plan = []

    def add(source, row, selection):
        key = row.get('task_id', f"{row['label']}:{row['idx']}")
        plan.append({
            'audit_id':f'{source}:{key}',
            'source_path':str((BASE/source/'samples.jsonl').relative_to(ROOT)),
            'source_protocol_path':str((BASE/source/'protocol.json').relative_to(ROOT)),
            'source_protocol_id':row['protocol_id'],
            'source_record_sha256':hashlib.sha256(json.dumps(row, sort_keys=True,
                separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
            'selection':selection, 'label':row['label'], 'idx':row['idx'],
            'suite':row.get('suite', 'fresh' if 'fresh_' in source else 'archived'),
            'condition':row.get('condition', 'baseline'),
            'attacker':row.get('attacker', 'frozen'),
            'decoder_seed':row.get('decoder_seed', row.get('reconstruction_seed')),
            'first_input':row['actual_encoder_inputs'][0],
            'saved_first_prediction':row['first_prediction'],
            'reference':row['reference'], 'samples':3})

    for source in source_names[:3]:
        for label in sorted({r['label'] for r in sources[source]}):
            rows = sorted([r for r in sources[source] if r['label']==label],
                          key=lambda r:(r['actual_encoder_inputs'][0]['full_token_count'], r['idx']))
            add(source, rows[0], 'minimum first encoder token count within source/model; lowest idx tie')
            add(source, rows[-1], 'maximum first encoder token count within source/model; highest idx tie')
    source = source_names[-1]
    for i, condition in enumerate(['baseline', 'bucket_8', 'pad_32', 'batch_2', 'batch_4', 'rand_pad_8']):
        label = 'qwen_1_5b' if i % 2 == 0 else 'llama_3_2_3b'
        rows = sorted([r for r in sources[source] if r['suite']=='defense_pilot'
                       and r['condition']==condition and r['label']==label], key=lambda r:r['idx'])
        add(source, rows[0], 'six defense-pilot conditions; alternating Qwen/Llama; lowest idx')
    assert len(plan)==24
    return plan


if __name__ == '__main__':
    planned = build_inventory()
    saved = [json.loads(x) for x in (HERE/'inputs.jsonl').read_text().splitlines()]
    assert planned == saved, 'Frozen inventory differs from deterministic selection'
    print('PASS: 24/24 frozen audit inputs reproduced from original records')
