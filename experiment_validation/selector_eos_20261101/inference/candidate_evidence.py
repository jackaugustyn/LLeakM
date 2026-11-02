"""Dependency-free evidence checks used before committing or resuming records.

The campaign's separate verifier independently audits the same evidence. This
module does not run inference or alter generation state.
"""
from __future__ import annotations
import hashlib
import json
import math


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def record_digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric score')
    return float(value)


def validate_candidate_evidence(audit, samples, selected_text):
    if audit.get('schema') != 'eos-bounded-candidate-audit/1':
        raise ValueError('Missing or unrecognized candidate audit')
    if audit.get('selector') != 'processed-score-sum-through-first-eos':
        raise ValueError('Wrong selector evidence')
    if audit.get('samples_requested') != samples:
        raise ValueError('Candidate request count mismatch')
    candidates = audit.get('candidates', [])
    if len(candidates) != samples:
        raise ValueError('Candidate evidence incomplete')
    prefix = audit['decoder_prefix_length']
    if not isinstance(prefix, int) or prefix < 0:
        raise ValueError('Invalid decoder prefix length')
    eos_ids = audit['eos_token_ids']
    if not isinstance(eos_ids,list) or any(type(x) is not int for x in eos_ids):
        raise ValueError('EOS IDs missing')
    best = {}
    for i, candidate in enumerate(candidates):
        if candidate['candidate_index'] != i:
            raise ValueError('Candidate indices are not contiguous')
        ids = candidate['sequence_token_ids']
        generated = candidate['generated_token_ids']
        if generated != ids[prefix:] or any(type(x) is not int for x in ids):
            raise ValueError('Generated token alignment mismatch')
        eos = next((j for j, token in enumerate(generated) if token in eos_ids), None)
        if candidate['first_eos_generated_offset'] != eos:
            raise ValueError('First EOS offset mismatch')
        score = finite_number(candidate['confidence'])
        if samples == 1:
            if candidate['transition_scores'] is not None or candidate['scored_token_count'] != 0 or score != 0:
                raise ValueError('Single-candidate path must bypass scoring')
        else:
            scores = candidate['transition_scores']
            used = eos + 1 if eos is not None else len(generated)
            if len(scores) != len(generated) or candidate['scored_token_count'] != used:
                raise ValueError('Transition/EOS score extent mismatch')
            prefix_scores = [finite_number(x) for x in scores[:used]]
            # GPU float32 reduction and Python sum have small rounding differences.
            if not math.isclose(score, sum(prefix_scores), rel_tol=1e-6, abs_tol=1e-4):
                raise ValueError('Confidence does not equal the EOS-bounded score sum')
            for value in scores[used:]:
                if value not in ['-Infinity', 'Infinity', 'NaN']:
                    finite_number(value)
        text = candidate['text']
        if not isinstance(text, str) or candidate['eligible'] != bool(text):
            raise ValueError('Candidate eligibility/text mismatch')
        if text and (text not in best or score > best[text][1]):
            best[text] = (i, score)
    ranked = sorted(best, key=lambda text: best[text][1], reverse=True)
    indices = [best[text][0] for text in ranked]
    if audit['ranked_texts'] != ranked or audit['ranked_candidate_indices'] != indices:
        raise ValueError('Saved ordering/deduplication does not follow confidence')
    selected = indices[0] if indices else None
    if audit['selected_candidate_index'] != selected or selected_text != (ranked[0] if ranked else ''):
        raise ValueError('Selected candidate differs from the saved ranking')


def validate_record_candidates(record):
    entries = record.get('actual_encoder_inputs', [])
    if not entries:
        raise ValueError('Missing encoder/candidate evidence')
    for entry in entries:
        validate_candidate_evidence(entry.get('candidate_audit', {}), entry['samples'], entry['selected_text'])
    if record['first_prediction'] != entries[0]['selected_text']:
        raise ValueError('First prediction is not the first selected candidate')


def slim_secondary(record):
    """Keep aggregation fields; do not retain large candidate tensors in RAM."""
    fields = ['task_id', 'suite', 'label', 'condition', 'attacker', 'idx', 'seed_index',
              'metrics', 'observed_payload_overhead_pct', 'n_observed_units',
              'stopped_before_requested_segments', 'empty_first_prediction']
    return {key: record[key] for key in fields}
