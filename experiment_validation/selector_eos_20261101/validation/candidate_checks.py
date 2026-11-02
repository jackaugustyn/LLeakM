"""Independent CPU checks of saved candidate evidence; never imports the selector."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

FIRST = 'royweiss1/T5_FirstSentences'
MIDDLE = 'royweiss1/T5_MiddleSentences'
PREFIX_FIRST = 'Translate the Special Tokens to English. \nSpecial Tokens:'
PREFIX_MIDDLE = 'Translate the Special Tokens to English, given the context. \nContext: '
WORD = re.compile(r"[A-Za-z0-9']+")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def record_digest(row):
    return hashlib.sha256(packed({k:v for k,v in row.items() if k!='record_sha256'}).encode()).hexdigest()


def phi(reference, prediction):
    a, b = Counter(WORD.findall(reference.lower())), Counter(WORD.findall(prediction.lower()))
    den = math.sqrt(sum(x*x for x in a.values()) * sum(x*x for x in b.values()))
    return sum(x*b[k] for k,x in a.items())/den if den else 0.0


def number(value):
    if isinstance(value, str):
        require(value in {'NaN','Infinity','-Infinity'}, 'Unknown nonfinite score encoding')
        return float(value)
    require(isinstance(value, (int, float)) and not isinstance(value, bool), 'Score is not numeric')
    return float(value)


def sum_check(values, saved, where):
    values = [number(v) for v in values]
    saved = number(saved)
    if any(math.isnan(v) for v in values) or (math.inf in values and -math.inf in values):
        require(math.isnan(saved), where + ': expected NaN')
        return
    if any(math.isinf(v) for v in values):
        expected = -math.inf if -math.inf in values else math.inf
        require(saved == expected, where + ': infinity mismatch')
        return
    expected = math.fsum(values)
    # Saved scalars are CUDA float32 reductions, while this sum is double precision.
    # This bound is stated explicitly and does not replace exact rank/tie checks.
    tolerance = max(1e-5, 2e-6 * max(1., sum(abs(v) for v in values)))
    require(math.isfinite(saved) and abs(saved-expected) <= tolerance,
            f'{where}: prefix/full sum mismatch {saved} vs {expected} (tol={tolerance})')


def rank(candidates, field):
    best = {}
    for c in candidates:
        if not c['text']:
            continue
        value = number(c[field])
        # Dictionary position preserves the first appearance of a text even
        # when a later duplicate replaces its confidence and candidate index.
        if c['text'] not in best or value > best[c['text']][0]:
            best[c['text']] = (value, c['candidate_index'])
    return [(text, value[1]) for text, value in
            sorted(best.items(), key=lambda item:item[1][0], reverse=True)]


def validate_call(call, tokenizer, expected_samples, where='call'):
    a = call['candidate_audit']
    require(a['schema']=='eos-bounded-candidate-audit/1', where+': audit schema')
    require(a['selector']=='processed-score-sum-through-first-eos', where+': selector')
    require(a['samples_requested']==expected_samples==call['samples'], where+': candidate count')
    require(a['decoder_prefix_length']==1, where+': T5 decoder start count')
    require(a['eos_token_ids']==[tokenizer.eos_token_id], where+': pinned EOS ID')
    cs = a['candidates']
    require(len(cs)==expected_samples, where+': missing candidate')
    require(a['score_dtype']==('torch.float32' if expected_samples>1 else None), where+': score precision')
    counts = Counter(calls=1, candidates=len(cs), ranked_calls=int(expected_samples>1), single_candidate_calls=int(expected_samples==1))
    for i, c in enumerate(cs):
        loc = f'{where}/candidate-{i}'
        require(c['candidate_index']==i, loc+': index')
        seq, generated = c['sequence_token_ids'], c['generated_token_ids']
        require(seq and seq[0]==tokenizer.pad_token_id and len(seq)<=80, loc+': decoder start/limit')
        require(seq[1:]==generated, loc+': score alignment')
        require(all(type(v) is int and v>=0 for v in seq), loc+': token IDs')
        decoded = tokenizer.decode(seq, skip_special_tokens=True).strip()
        require(decoded==c['text'] and c['eligible']==bool(decoded), loc+': text/eligibility')
        eos = next((j for j,t in enumerate(generated) if t==tokenizer.eos_token_id), None)
        require(c['first_eos_generated_offset']==eos, loc+': EOS offset')
        if eos is not None:
            require(all(t==tokenizer.pad_token_id for t in generated[eos+1:]), loc+': non-PAD after EOS')
        if expected_samples==1:
            require(c['transition_scores'] is None and c['confidence']==0 and c['scored_token_count']==0
                    and c['legacy_full_step_confidence'] is None, loc+': singleton score bypass')
            continue
        scores = c['transition_scores']
        require(len(scores)==len(generated), loc+': transition count')
        n = eos+1 if eos is not None else len(generated)
        require(c['scored_token_count']==n, loc+': EOS-inclusive score count')
        require(all(math.isfinite(number(v)) for v in scores[:n]), loc+': nonfinite included score')
        require(all(number(v)<=1e-6 for v in scores[:n]), loc+': positive normalized log score')
        sum_check(scores[:n], c['confidence'], loc+'/eos-sum')
        sum_check(scores, c['legacy_full_step_confidence'], loc+'/legacy-sum')
        counts['scored_tokens'] += n
        if n < len(scores):
            counts['candidates_with_post_eos_tail'] += 1
            counts['excluded_post_eos_tokens'] += len(scores)-n
            counts['candidates_with_nonfinite_post_eos_tail'] += any(not math.isfinite(number(v)) for v in scores[n:])
    current = rank(cs, 'confidence')
    require(a['ranked_texts']==[t for t,i in current], where+': exact ranked texts')
    require(a['ranked_candidate_indices']==[i for t,i in current], where+': exact ranked indices')
    index = current[0][1] if current else None
    text = current[0][0] if current else ''
    require(a['selected_candidate_index']==index and call['selected_text']==text, where+': selected candidate')
    legacy = rank(cs, 'legacy_full_step_confidence') if expected_samples>1 else current
    legacy_text = legacy[0][0] if legacy else ''
    counts['same_candidate_pool_selection_changes'] = int(text!=legacy_text)
    return dict(counts), legacy_text


def segments(lengths):
    cuts = [-1]+[i for i,x in enumerate(lengths) if x==1]
    if cuts[-1]!=len(lengths)-1:
        cuts.append(len(lengths)-1)
    parts = [lengths[a+1:b+1] for a,b in zip(cuts,cuts[1:])]
    for i in range(len(parts)-1):
        if parts[i][-3:]==[3,1,1]:
            parts[i+1]=parts[i][-3:]+parts[i+1]
            parts[i]=parts[i][:-3]
    out, pending = [], []
    for part in filter(None, parts):
        pending += part
        if len(pending)>=10:
            out.append(pending); pending=[]
    if pending:
        out.append(pending)
    return out


def concat(parts):
    return re.sub(r'\s+([,.;:!?])', r'\1', ' '.join(x.strip() for x in parts if x.strip()))


def audit_record(row, source, original, tokenizers, *, secondary=False):
    """Check one complete saved record and its original-selector counterpart."""
    key = row.get('task_id', str((row['label'],row['idx'])))
    require(row.get('record_sha256')==record_digest(row), key+': new record digest')
    for field in ['reference','reference_sha256','generation_log_sha256','run_id','label','idx']:
        require(row[field]==source[field]==original[field], key+': fixed source '+field)
    seed_field = 'decoder_seed' if secondary else 'reconstruction_seed'
    require(row[seed_field]==source[seed_field]==original[seed_field], key+': fixed seed')
    length_field = 'attacker_input_lengths' if secondary else 'token_lengths'
    require(row[length_field]==source[length_field]==original[length_field], key+': fixed lengths')
    if secondary:
        require(all(row[k]==v for k,v in source.items()), key+': frozen inventory differs')
        prior_phi = original['metrics']['phi']
        prior_digest = original['record_sha256']
        require(record_digest(original)==prior_digest, key+': original secondary digest')
        samples = source['samples_per_segment']
        max_segments = source['max_sentences']
    else:
        prior_phi = original['phi']
        prior_digest = hashlib.sha256(packed(original).encode()).hexdigest()
        samples, max_segments = 3, 0
    require(row['original_selector_prediction']==original['prediction'], key+': original output binding')
    require(row['original_selector_phi']==prior_phi and row['original_selector_record_sha256']==prior_digest,
            key+': original score/hash binding')
    require(abs(phi(row['reference'],row['prediction'])-(row['metrics']['phi'] if secondary else row['phi']))<1e-12,
            key+': recomputed phi')
    require(abs(phi(original['reference'],original['prediction'])-prior_phi)<1e-12, key+': original phi')
    ss = segments(row[length_field]); requested=min(max_segments,len(ss)) if max_segments else len(ss)
    calls = row['actual_encoder_inputs']
    require(row['available_segments']==len(ss), key+': segment count')
    require(0<len(calls)<=requested, key+': attempted segment scope')
    counts, chosen = Counter(records=1), []
    for i, call in enumerate(calls):
        model = FIRST if i==0 else MIDDLE
        require(call['model_id']==model and call['max_length']==80, key+': model/limit')
        encoded_lengths = ''.join(' _'+str(n) for n in ss[i])
        expected = PREFIX_FIRST+encoded_lengths if i==0 else PREFIX_MIDDLE+chosen[-1]+' \nSpecial Tokens:'+encoded_lengths
        require(call['text']==expected, key+': serialization/context chain')
        tk = tokenizers[model]
        full = tk([expected],padding=True,truncation=False)['input_ids'][0]
        kept = tk([expected],padding=True,truncation=True,max_length=80)['input_ids'][0]
        require(call['retained_token_ids']==kept and call['full_token_count']==len(full), key+': encoder tokenization')
        if 'full_token_ids' in call:
            require(call['full_token_ids']==full, key+': full encoder IDs')
        require(call['truncated']==(len(full)>len(kept)), key+': truncation flag')
        evidence, legacy_text = validate_call(call,tk,samples,f'{key}/segment-{i}')
        counts.update(evidence)
        if i==0:
            require(legacy_text==original['first_prediction'], key+': first-call legacy counterfactual differs from original')
            counts['first_call_original_selection_matches'] += 1
            counts['first_call_original_selection_matches_multicandidate'] += samples>1
            counts['first_call_selection_changes'] += call['selected_text']!=original['first_prediction']
        chosen.append(call['selected_text'])
        if i>0 and not chosen[-1]:
            require(i==len(calls)-1, key+': continued after empty middle selection')
    require(len(calls)==requested or (len(calls)>=2 and not chosen[-1]), key+': unexplained early stop')
    completed=1+sum(bool(x) for x in chosen[1:])
    require(row['reconstructed_segments']==completed, key+': selected segment count')
    require(row['first_prediction']==chosen[0] and row['prediction']==concat(chosen), key+': assembled output')
    if secondary and row['suite']=='map':
        require(row['prediction']==original['prediction'], key+': MAP negative control changed')
        counts['map_predictions_matched'] += 1
    counts['full_response_prediction_changes'] += row['prediction']!=original['prediction']
    return dict(counts)
