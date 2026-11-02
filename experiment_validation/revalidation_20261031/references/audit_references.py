"""Audit archived decoding evidence without generating or repairing references."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
HUB = Path.home() / '.cache/huggingface/hub'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {
        'scope': 'Archived token text and IDs inventory plus local tokenizer inversion; no model generation, no heuristic reference repair.',
        'historical_tokenizer_revision_attested': False,
        'models': {},
    }
    gemma_records = []
    for samples in sorted((ROOT / 'experiment_validation/results').glob('run_full_20260920_*/samples.jsonl')):
        rows = [json.loads(x) for x in samples.read_text().splitlines() if x.strip()]
        counts = Counter()
        aggregate = hashlib.sha256()
        reference_examples = []
        for row in rows:
            counts['samples'] += 1
            path = ROOT / 'logs' / (row['run_id'] + '.json')
            if not path.is_file():
                counts['missing_logs'] += 1
                continue
            data = json.loads(path.read_text())
            aggregate.update(path.name.encode() + b'\0' + bytes.fromhex(sha(path)))
            steps = data['steps']
            counts['logs'] += 1
            counts['generated_token_ids_in_logs'] += 'generated_token_ids' in data
            counts['step_token_ids_in_logs'] += sum('token_id' in s for s in steps)
            counts['steps'] += len(steps)
            text = row['response_text']
            counts['references_without_ASCII_spaces'] += ' ' not in text
            counts['references_containing_UFFFD'] += '\ufffd' in text
            counts['reference_log_mismatch'] += data['response_text'] != text
            counts['reference_fragment_concat_mismatch'] += ''.join(s['token_text'] for s in steps) != text
            counts['logged_UTF8_length_mismatch'] += sum(len(s['token_text'].encode()) != s['token_utf8_len'] for s in steps)
            if (' ' not in text or '\ufffd' in text) and len(reference_examples) < 2:
                reference_examples.append({'run_id': row['run_id'], 'excerpt': text[:200]})
            if 'gemma' in samples.parent.name:
                gemma_records.append((row, steps))
        model_id = rows[0]['victim_model_id']
        cached = list((HUB / ('models--' + model_id.replace('/', '--'))).glob('snapshots/*/tokenizer.json'))
        result['models'][samples.parent.name] = {
            'model_id': model_id,
            'sample_sha256': sha(samples),
            'ordered_log_hashes_sha256': aggregate.hexdigest(),
            'counts': dict(counts),
            'cached_tokenizer_json': [str(p) for p in cached],
            'examples': reference_examples,
            'exact_ID_recovery_from_supplied_records': 'unavailable: token IDs absent; concatenated text is not evidence of a unique inverse',
        }
    print('Archived inventories complete.', flush=True)
    gemma_files = list((HUB / 'models--google--gemma-2-2b-it').glob('snapshots/*/tokenizer.json'))
    if gemma_files:
        file = gemma_files[0]
        tokenizer = AutoTokenizer.from_pretrained(str(file.parent), use_fast=True, local_files_only=True)
        observed = {s['token_text'] for _, steps in gemma_records for s in steps}
        inverse = defaultdict(list)
        for token_id in range(len(tokenizer)):
            decoded = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if decoded in observed:
                inverse[decoded].append(token_id)
        counts = Counter()
        examples = []
        for row, steps in gemma_records:
            ambiguous = [i for i, s in enumerate(steps) if len(inverse[s['token_text']]) > 1]
            missing = [i for i, s in enumerate(steps) if not inverse[s['token_text']]]
            counts['samples'] += 1
            counts['samples_with_ambiguous_token_inverse'] += bool(ambiguous)
            counts['samples_with_unmatched_token_inverse'] += bool(missing)
            counts['ambiguous_steps'] += len(ambiguous)
            counts['unmatched_steps'] += len(missing)
            if not missing:
                first_ids = [inverse[s['token_text']][0] for s in steps]
                last_ids = [inverse[s['token_text']][-1] for s in steps]
                for tag, ids in [('first_candidate', first_ids), ('last_candidate', last_ids)]:
                    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    counts[tag + '_full_decode_matches_reference'] += decoded == row['response_text']
                counts['distinct_first_last_ID_sequences'] += first_ids != last_ids
            if not ambiguous and not missing:
                ids = [inverse[s['token_text']][0] for s in steps]
                full = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                counts['uniquely_invertible_under_local_tokenizer'] += 1
                counts['unique_inverse_full_decode_differs_from_archived_reference'] += full != row['response_text']
            elif len(examples) < 3:
                positions = ambiguous[:2]
                examples.append({'run_id': row['run_id'], 'ambiguous': [
                    {'step': i, 'token_text': steps[i]['token_text'], 'candidate_token_ids': inverse[steps[i]['token_text']][:20],
                     'candidate_count': len(inverse[steps[i]['token_text']])}
                    for i in positions], 'unmatched_positions': missing[:5]})
        result['gemma_local_inverse'] = {
            'tokenizer_json': str(file), 'tokenizer_json_sha256': sha(file),
            'tokenizer_class': type(tokenizer).__name__,
            'vocabulary_size': len(tokenizer), 'counts': dict(counts), 'examples': examples,
            'interpretation': 'Inversion is conditional on the local tokenizer, not historical attestation. First/last candidate choices test two compatible sequences, not all possible combinations. Distinct IDs may decode to the same reference; inverse ambiguity is not by itself a reference defect. No inferred token IDs are promoted to ground truth or saved as repaired records.',
        }
        sys.path.insert(0, str(ROOT))
        from app import _decode_token_fragments
        cases = ["Hello world.\n\nA second line.", "Żółć, café, Ελληνικά, 中文, 🙂", "A \U0001fae8 B", ""]
        checks = []
        for text in cases:
            ids = tokenizer.encode(text, add_special_tokens=False)
            fragments, full = _decode_token_fragments(tokenizer, ids)
            individual = ''.join(tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids)
            assert ''.join(fragments) == full
            checks.append({'input': text, 'generated_ID_count': len(ids), 'fragment_count': len(fragments),
                           'empty_fragments': sum(not s for s in fragments),
                           'contextual_equals_full_decode': ''.join(fragments) == full,
                           'individual_decode_equals_full_decode': individual == full})
        result['new_decoder_checks_with_cached_gemma'] = checks
    (OUT / 'reference_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v['counts'] for k, v in result['models'].items()}, indent=2))
    print(json.dumps(result.get('gemma_local_inverse', {}).get('counts'), indent=2))


if __name__ == '__main__':
    main()
