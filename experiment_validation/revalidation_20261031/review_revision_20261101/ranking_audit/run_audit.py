"""Bounded first-segment replay of the frozen three-candidate T5 selector.

Run with system python3 in the original environment; no original data/code writes.
The input inventory is frozen before generation and selected without candidate
outcomes. This purposefully stratified sample is not a prevalence estimator.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['LLEAKM_RECONSTRUCT_DEVICE'] = 'cuda'

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import weiss_reconstruction as wr
from eos_scoring import scores_through_eos


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def portable_float(value):
    value = float(value)
    if math.isnan(value):
        return 'NaN'
    if math.isinf(value):
        return '-Infinity' if value < 0 else 'Infinity'
    return value


def rank_candidates(texts, scores):
    """Mirror frozen deduplication and stable sorting; change only the scores."""
    best = {}
    for text, score in zip(texts, scores):
        if not text:
            continue
        if text not in best or score > best[text]:
            best[text] = score
    return sorted(best.items(), key=lambda item: item[1], reverse=True)


def main():
    inputs = [json.loads(x) for x in (HERE/'inputs.jsonl').read_text().splitlines()]
    if not torch.cuda.is_available():
        raise RuntimeError('Original CUDA device required; no CPU fallback')
    if torch.__version__ != '2.5.1+cu124' or transformers.__version__ != '4.46.3':
        raise RuntimeError('Pinned environment differs')
    if digest(ROOT/'weiss_reconstruction.py') != '55ebb745e9cafd08d89931f17ef26a5858ff8884c526e16aa2b0201784c83dc1':
        raise RuntimeError('Historical selector source changed')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_id = wr.FIRST_SENTENCES_MODEL
    revision = '057cf090b719f5cf89da504babd2122d404e0953'
    checkpoint_files = {
        name: digest(hf_hub_download(model_id, name, revision=revision, local_files_only=True))
        for name in ['pytorch_model.bin', 'tokenizer.json', 'config.json']
    }
    source_paths = sorted({x['source_path'] for x in inputs} | {x['source_protocol_path'] for x in inputs})
    for source in {x['source_protocol_path'] for x in inputs}:
        protocol = json.loads((ROOT/source).read_text())
        stored_hashes = protocol.get('checkpoint_file_sha256', protocol.get('checkpoint_files_sha256'))[model_id]
        if any(stored_hashes[name] != value for name, value in checkpoint_files.items()):
            raise ValueError('Source model hashes differ from cached files')
    protocol = {
        'schema': 'bounded-first-segment-selector-audit/1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': '24 first-segment cases selected by input-length extrema and pilot condition before inference; 72 candidates. No estimate of full-campaign prevalence or full-response effect.',
        'selection': 'Two input-length extrema per seven archived and two fresh cohorts; six defense-pilot conditions, alternating models, lowest prompt index. Extremal ties use lowest/highest idx.',
        'inputs_sha256': digest(HERE/'inputs.jsonl'),
        'script_sha256': digest(__file__),
        'eos_scoring_sha256': digest(HERE/'eos_scoring.py'),
        'reconstructor_sha256': digest(ROOT/'weiss_reconstruction.py'),
        'source_sha256': {p:digest(ROOT/p) for p in source_paths},
        'model_id': model_id, 'revision': revision, 'checkpoint_file_sha256': checkpoint_files,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'threads': 4, 'precision':'float32',
        'candidate_count':3, 'max_length':80,
        'generation_settings': {'no_repeat_ngram_size':2, 'do_sample':True, 'top_k':50, 'temperature':1.0},
        'score_meaning':'Normalized processed sampling distributions from compute_transition_scores; these include top-k and no-repeat-ngram processing. All selected-token transition scores are retained; full-vocabulary logits are not stored.',
        'eos_rule':'Align transition columns to the generated sequence suffix; include the first EOS, exclude every subsequent token; retain all steps when EOS is absent.',
        'nonfinite_encoding':'Nonfinite values use strings -Infinity, Infinity, NaN; no numeric sentinel.',
        'source_replay_check':'Actual frozen _sample_and_rank is called unchanged; its generated candidates and transitions are intercepted, and selected text is checked against saved first_prediction.',
    }
    if (HERE/'results.jsonl').exists():
        raise RuntimeError('Results already exist; refusing overwrite')
    write_json(HERE/'protocol.json', protocol)
    start = time.monotonic()
    write_json(HERE/'status.json', {'status':'loading_model','expected':len(inputs),'completed':0})
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, revision=revision,
        local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float32)
    model.to('cuda').eval()
    generate = model.generate
    compute = model.compute_transition_scores
    captured = {}

    def observed_generate(**kwargs):
        output = generate(**kwargs)
        captured['sequences'] = output.sequences.detach().cpu()
        return output

    def observed_compute(sequences, scores, **kwargs):
        transition = compute(sequences, scores, **kwargs)
        captured['transitions'] = transition.detach().cpu()
        captured['legacy_sums'] = torch.stack([row.sum() for row in transition]).detach().cpu()
        eos_result = scores_through_eos(sequences, transition, model.generation_config.eos_token_id)
        captured['eos_scores'] = eos_result[0].detach().cpu()
        captured['valid_counts'], captured['eos_offsets'] = eos_result[1:]
        return transition

    model.generate = observed_generate
    model.compute_transition_scores = observed_compute
    eos = model.generation_config.eos_token_id
    done = []
    for row in inputs:
        retained = tokenizer([row['first_input']['text']], max_length=80, padding=True, truncation=True)['input_ids'][0]
        if retained != row['first_input']['retained_token_ids']:
            raise ValueError('Input token IDs differ from saved first segment')
        wr.seed_reconstructor(row['decoder_seed'])
        captured.clear()
        t = time.monotonic()
        old_ranked = wr._sample_and_rank(model, tokenizer, 'cuda', row['first_input']['text'], samples=3, max_length=80)
        sequences, transitions = captured['sequences'], captured['transitions']
        fixed_scores = captured['eos_scores']
        valid_counts, eos_offsets = captured['valid_counts'], captured['eos_offsets']
        texts = [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in sequences]
        legacy_scores = captured['legacy_sums']
        independent_old = rank_candidates(texts, legacy_scores.tolist())
        if old_ranked != independent_old:
            raise AssertionError('Independent historical score replay disagrees')
        corrected_ranked = rank_candidates(texts, fixed_scores.tolist())
        old_text = old_ranked[0][0] if old_ranked else ''
        fixed_text = corrected_ranked[0][0] if corrected_ranked else ''
        generated_offset = sequences.shape[1] - transitions.shape[1]
        candidates = []
        for i in range(len(texts)):
            count = valid_counts[i]
            # An independently enumerated finite-prefix sum checks the utility.
            direct = sum(float(x) for x in transitions[i, :count])
            if math.isfinite(direct) and not math.isclose(direct, fixed_scores[i].item(), abs_tol=2e-5, rel_tol=2e-6):
                raise AssertionError('Independent EOS prefix sum disagrees')
            candidates.append({'candidate_index':i,'text':texts[i],
                'sequence_ids':sequences[i].tolist(),
                'generated_token_ids':sequences[i, generated_offset:].tolist(),
                'transition_scores':[portable_float(x) for x in transitions[i]],
                'first_eos_generated_offset':eos_offsets[i],
                'score_token_count_through_eos':count,
                'tail_token_count':transitions.shape[1]-count,
                'tail_pad_count':int((sequences[i,generated_offset+count:]==model.generation_config.pad_token_id).sum()),
                'legacy_score':portable_float(legacy_scores[i]),
                'eos_score':portable_float(fixed_scores[i]),
                'prefix_nonfinite_count':int((~torch.isfinite(transitions[i,:count])).sum()),
                'tail_nonfinite_count':int((~torch.isfinite(transitions[i,count:])).sum())})
        result = {'audit_id':row['audit_id'],'suite':row['suite'],'label':row['label'],
            'idx':row['idx'],'condition':row['condition'],'decoder_seed':row['decoder_seed'],
            'input_retained_ids':retained,'input_truncated':row['first_input']['truncated'],
            'saved_first_prediction':row['saved_first_prediction'],
            'legacy_selected_text':old_text,'eos_selected_text':fixed_text,
            'legacy_replay_matches_saved':old_text==row['saved_first_prediction'],
            'selected_text_changes':old_text!=fixed_text,
            'legacy_ranked':[[x,portable_float(y)] for x,y in old_ranked],
            'eos_ranked':[[x,portable_float(y)] for x,y in corrected_ranked],
            'decoder_prefix_token_count':generated_offset,'eos_token_id':eos,
            'pad_token_id':model.generation_config.pad_token_id,'candidates':candidates,
            'elapsed_seconds':time.monotonic()-t}
        with (HERE/'results.jsonl').open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            f.flush(); os.fsync(f.fileno())
        done.append(result)
        write_json(HERE/'status.json', {'status':'running','completed':len(done),
            'expected':len(inputs),'elapsed_seconds':time.monotonic()-start,
            'old_matches':sum(r['legacy_replay_matches_saved'] for r in done),
            'selection_changes':sum(r['selected_text_changes'] for r in done)})
        print(f"{len(done)}/{len(inputs)} {row['suite']} {row['label']} idx={row['idx']} old_match={result['legacy_replay_matches_saved']} changed={result['selected_text_changes']} seconds={result['elapsed_seconds']:.2f}", flush=True)
    model.generate = generate
    model.compute_transition_scores = compute
    candidates = [c for r in done for c in r['candidates']]
    summary = {'complete':True,'cases':len(done),'candidates':len(candidates),
        'legacy_matches_saved':sum(r['legacy_replay_matches_saved'] for r in done),
        'selected_text_changes':sum(r['selected_text_changes'] for r in done),
        'candidates_with_post_eos_tail':sum(c['tail_token_count']>0 for c in candidates),
        'candidates_with_nonfinite_post_eos_tail':sum(c['tail_nonfinite_count']>0 for c in candidates),
        'candidates_with_nonfinite_prefix':sum(c['prefix_nonfinite_count']>0 for c in candidates),
        'cases_by_suite':dict(Counter(r['suite'] for r in done)),
        'changes_by_suite':dict(Counter(r['suite'] for r in done if r['selected_text_changes'])),
        'elapsed_seconds':time.monotonic()-start,
        'scope':'Descriptive counts for the 24 selected first-segment cases only; no full-response metrics and no campaign-prevalence extrapolation.'}
    write_json(HERE/'summary.json', summary)
    write_json(HERE/'status.json', dict(summary,status='completed'))
    print(json.dumps(summary, indent=2),flush=True)


if __name__ == '__main__':
    main()
