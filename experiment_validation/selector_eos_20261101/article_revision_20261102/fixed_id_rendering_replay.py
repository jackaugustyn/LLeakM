"""CPU-only independent fixed-ID rendering replay bound to current EOS records."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
ROOT = CAMPAIGN.parents[1]
REFERENCES = ROOT / 'experiment_validation/revalidation_20261031/fresh_references'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def textsha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def current_bindings(path):
    return {(r['label'], r['idx']): {key: r[key] for key in
            ('run_id', 'reference', 'reference_sha256', 'token_lengths',
             'reconstruction_seed', 'protocol_id', 'record_sha256')}
            for r in read_rows(path)}


def main():
    main_path = CAMPAIGN / 'inference/cuda_full/samples.jsonl'
    archived_arm = current_bindings(main_path)
    report = {
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS',
        'scope': '600 recorded victim ID sequences replayed on CPU with pinned tokenizers; both rendering arms bound to current EOS campaign reconstructions. No victim or T5 inference and no earlier reconstruction-score comparison.',
        'method': 'Independently decode each fixed token ID and concatenate for the legacy arm; decode the same entire ID sequence for the full-sequence arm. Validate trace lengths, emitted fragments, references, input bindings and paired reconstruction seeds.',
        'legacy_call': 'tokenizer.decode([tok_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)',
        'full_sequence_call': 'tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)',
        'models': [],
        'source_sha256': {'current_archived_arm': sha(main_path)},
        'limits': [
            'Historical victim logs do not record original token IDs. Output equality does not recover those IDs or independently attest historical checkpoint versions.',
            'Within each recorded sequence the rendering contrast isolates a change in emitted text and byte lengths; the reconstruction-score contrast jointly changes the trace and scoring reference.',
            'The two verified-reference cohorts remain separate from the seven-model archived-reference ranking.',
        ],
    }
    records = []
    for dirname, cohort in [('tinyllama', 'fresh_tinyllama_cuda'), ('phi', 'fresh_phi_cuda')]:
        folder = REFERENCES / dirname
        source_path = folder / 'canonical_traces.jsonl'
        rows = list(read_rows(source_path))
        assert len(rows) == 300 and len({r['idx'] for r in rows}) == 300
        protocol = json.loads((folder / 'protocol.json').read_text())
        model_id, revision = protocol['model_id'], protocol['revision']
        cache = Path.home() / '.cache/huggingface/hub' / ('models--' + model_id.replace('/', '--')) / 'snapshots' / revision
        for filename in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
            assert sha(cache / filename) == protocol['checkpoint_file_sha256'][filename]
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, use_fast=True)
        current_path = CAMPAIGN / 'inference' / cohort / 'samples.jsonl'
        verified_arm = current_bindings(current_path)
        counts = Counter()
        decoded = {}
        for row in rows:
            key = (row['label'], row['idx'])
            old = archived_arm[key]
            fresh = verified_arm[key]
            fresh_log_path = ROOT / row['generation_log_path']
            assert sha(fresh_log_path) == row['generation_log_sha256']
            fresh_log = json.loads(fresh_log_path.read_text())
            archived_log = json.loads((ROOT / 'logs' / (row['historical_run_id'] + '.json')).read_text())
            assert old['run_id'] == row['historical_run_id']
            assert fresh['run_id'] == row['run_id']
            assert old['reconstruction_seed'] == fresh['reconstruction_seed']
            assert archived_log['prompt'] == fresh_log['prompt'] == row['prompt']
            ids = row['generated_token_ids']
            assert ids == fresh_log['generated_token_ids'] == [s['token_id'] for s in fresh_log['steps']]
            full = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            for token_id in set(ids):
                if token_id not in decoded:
                    decoded[token_id] = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            legacy_fragments = [decoded[token_id] for token_id in ids]
            legacy = ''.join(legacy_fragments)
            legacy_lengths = [len(s.encode('utf-8')) for s in legacy_fragments]
            full_fragments = [s['token_text'] for s in fresh_log['steps']]
            full_lengths = [len(s.encode('utf-8')) for s in full_fragments]
            checks = {
                'legacy_text_equals_archived_emitted_text': legacy == archived_log['response_text'],
                'legacy_fragments_equal_archived_fragments': legacy_fragments == [s['token_text'] for s in archived_log['steps']],
                'legacy_lengths_equal_archived_lengths': legacy_lengths == [s['token_utf8_len'] for s in archived_log['steps']],
                'legacy_text_equals_current_archived_arm_reference': legacy == old['reference'],
                'legacy_lengths_equal_current_archived_arm_inputs': legacy_lengths == old['token_lengths'],
                'legacy_reference_hash_matches_current_archived_arm': textsha(legacy) == old['reference_sha256'],
                'full_text_equals_verified_reference_and_current_arm': full == row['reference'] == fresh_log['response_text'] == fresh['reference'],
                'full_fragments_concatenate_to_full_text': ''.join(full_fragments) == full,
                'full_lengths_equal_verified_trace_and_current_inputs': full_lengths == row['token_lengths'] == fresh['token_lengths'],
                'full_reference_hash_matches_current_arm': textsha(full) == fresh['reference_sha256'],
                'paired_reconstruction_seeds_equal': old['reconstruction_seed'] == fresh['reconstruction_seed'],
            }
            assert all(checks.values()), (key, checks)
            for name, value in checks.items():
                counts[name] += value
            counts['records'] += 1
            counts['generated_tokens'] += len(ids)
            counts['legacy_references_without_ascii_spaces'] += ' ' not in legacy
            counts['full_references_without_ascii_spaces'] += ' ' not in full
            counts['legacy_and_full_equal_after_removing_ascii_spaces'] += legacy.replace(' ', '') == full.replace(' ', '')
            counts['legacy_and_full_identical_text'] += legacy == full
            changed = sum(a != b for a, b in zip(legacy_fragments, full_fragments))
            counts['changed_fragment_positions'] += changed
            counts['changed_positions_are_restored_leading_ascii_space'] += sum(b == ' ' + a for a, b in zip(legacy_fragments, full_fragments) if a != b)
            records.append({'label': row['label'], 'idx': row['idx'], 'token_count': len(ids),
                            'reconstruction_seed': old['reconstruction_seed'],
                            'legacy_reference_sha256': textsha(legacy), 'full_reference_sha256': textsha(full),
                            'current_archived_arm_record_sha256': old['record_sha256'],
                            'current_verified_arm_record_sha256': fresh['record_sha256'],
                            'archived_protocol_id': old['protocol_id'], 'verified_protocol_id': fresh['protocol_id'],
                            'checks': checks, 'changed_fragment_positions': changed})
        report['models'].append({'label': rows[0]['label'], 'model_id': model_id,
                                 'tokenizer_revision': revision, 'tokenizer_sha256': sha(cache / 'tokenizer.json'),
                                 'n': len(rows), 'counts': dict(counts)})
        report['source_sha256'][dirname + '_recorded_ID_inventory'] = sha(source_path)
        report['source_sha256'][dirname + '_protocol'] = sha(folder / 'protocol.json')
        report['source_sha256'][dirname + '_current_verified_arm'] = sha(current_path)
    report['records_checked'] = len(records)
    report['all_600_current_arm_bindings_pass'] = len(records) == 600 and all(all(r['checks'].values()) for r in records)
    assert report['all_600_current_arm_bindings_pass']
    report['script_sha256'] = sha(Path(__file__))
    (HERE / 'fixed_id_rendering_replay_records.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    report['records_file_sha256'] = sha(HERE / 'fixed_id_rendering_replay_records.jsonl')
    (HERE / 'fixed_id_rendering_replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'records_checked': len(records), 'models': report['models']}, indent=2))


if __name__ == '__main__':
    main()
