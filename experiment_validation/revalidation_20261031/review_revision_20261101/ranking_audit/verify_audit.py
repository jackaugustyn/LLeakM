"""Read-only independent check of saved candidate IDs, scores and selection."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def number(x):
    return float(x)


def rerank(candidates, field):
    scores = {}
    for c in candidates:
        if not c['text']:
            continue
        value = number(c[field])
        if c['text'] not in scores or value > scores[c['text']]:
            scores[c['text']] = value
    return sorted(scores, key=scores.__getitem__, reverse=True)


def main():
    inputs = [json.loads(x) for x in (HERE/'inputs.jsonl').read_text().splitlines()]
    rows = [json.loads(x) for x in (HERE/'results.jsonl').read_text().splitlines()]
    summary = json.loads((HERE/'summary.json').read_text())
    protocol = json.loads((HERE/'protocol.json').read_text())
    assert hashlib.sha256((HERE/'inputs.jsonl').read_bytes()).hexdigest()==protocol['inputs_sha256']
    assert len(inputs)==len(rows)==24
    for source, row in zip(inputs, rows):
        assert source['audit_id']==row['audit_id']
        assert source['decoder_seed']==row['decoder_seed']
        assert source['first_input']['retained_token_ids']==row['input_retained_ids']
        assert source['saved_first_prediction']==row['legacy_selected_text']
        assert len(row['candidates'])==3
        eos = row['eos_token_id']
        eos = [eos] if isinstance(eos, int) else eos
        for c in row['candidates']:
            ids = c['sequence_ids'][row['decoder_prefix_token_count']:]
            scores = [number(x) for x in c['transition_scores']]
            assert ids==c['generated_token_ids'] and len(ids)==len(scores)
            first = next((i for i,t in enumerate(ids) if t in eos), None)
            valid = first+1 if first is not None else len(ids)
            assert first==c['first_eos_generated_offset']
            assert valid==c['score_token_count_through_eos']
            assert len(ids)-valid==c['tail_token_count']
            assert sum(t==row['pad_token_id'] for t in ids[valid:])==c['tail_pad_count']
            assert sum(not math.isfinite(x) for x in scores[:valid])==c['prefix_nonfinite_count']
            assert sum(not math.isfinite(x) for x in scores[valid:])==c['tail_nonfinite_count']
            for value, recorded in [(sum(scores),c['legacy_score']), (sum(scores[:valid]),c['eos_score'])]:
                recorded = number(recorded)
                assert value==recorded or math.isclose(value,recorded,abs_tol=2e-5,rel_tol=2e-6)
        legacy, fixed = rerank(row['candidates'],'legacy_score'), rerank(row['candidates'],'eos_score')
        assert legacy==[x[0] for x in row['legacy_ranked']]
        assert fixed==[x[0] for x in row['eos_ranked']]
        assert (legacy[0] if legacy else '')==row['legacy_selected_text']
        assert (fixed[0] if fixed else '')==row['eos_selected_text']
        assert row['selected_text_changes']==(row['legacy_selected_text']!=row['eos_selected_text'])
    counts = {'legacy_matches_saved':sum(r['legacy_replay_matches_saved'] for r in rows),
              'selected_text_changes':sum(r['selected_text_changes'] for r in rows)}
    for key,value in counts.items():
        assert summary[key]==value
    result = {'status':'PASS', 'cases':len(rows), 'candidates':sum(len(r['candidates']) for r in rows),
              **counts, 'checks':['Saved input identity and seed', '24 saved first selections reproduced',
              'EOS boundary determined independently from token IDs', 'All prefix and full sums independently recomputed',
              'Post-EOS PAD and nonfinite counts', 'Stable text deduplication and both rankings'],
              'floating_point_tolerance':'abs_tol=2e-5, rel_tol=2e-6 for CPU Python sum versus recorded per-row CUDA sums'}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
