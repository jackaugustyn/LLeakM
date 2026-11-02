"""Revalidate every stored-text result without model inference or overwriting archives.

Run from any directory with PYTHONHASHSEED=0 python3 <this file>.
The result validates calculations conditional on supplied text, not acquisition.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import sys

import numpy as np
import scipy
from scipy import stats as spstats

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SRC = ROOT / 'experiment_validation/scripts'
OLD = ROOT / 'experiment_validation/analysis'
RERUN = OUT / 'analysis_rerun'
RERUN.mkdir(exist_ok=True)
WORD = re.compile(r"[A-Za-z0-9']+")
SEED = 20260706
BOOT = 10000


def readrows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def module(name):
    spec = importlib.util.spec_from_file_location('revalidation_' + name, SRC / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def levenshtein(a, b):
    """Exact bit-vector edit distance over Unicode code points (unit costs)."""
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return len(b)
    masks = defaultdict(int)
    for i, c in enumerate(a):
        masks[c] |= 1 << i
    vp, vn, score, last = ~0, 0, len(a), 1 << (len(a) - 1)
    for c in b:
        x = masks[c] | vn
        d0 = (((x & vp) + vp) ^ vp) | x
        hp = vn | ~(d0 | vp)
        hn = d0 & vp
        score += bool(hp & last) - bool(hn & last)
        hp = (hp << 1) | 1
        vn = d0 & hp
        vp = (hn << 1) | ~(d0 | hp)
    return score


def lcs(a, b):
    masks = defaultdict(int)
    for i, x in enumerate(b):
        masks[x] |= 1 << i
    state = 0
    for x in a:
        v = masks[x] | state
        state = v & ~(v - ((state << 1) | 1))
    return state.bit_count()


def metric(ref, pred):
    a, b = WORD.findall(ref.lower()), WORD.findall(pred.lower())
    ca, cb = Counter(a), Counter(b)
    denom = math.sqrt(sum(x*x for x in ca.values())) * math.sqrt(sum(x*x for x in cb.values()))
    phi = sum(ca[k] * cb[k] for k in ca) / denom if denom else 0.0
    overlap = sum((ca & cb).values())
    common = lcs(a, b)
    result = {'phi': phi, 'ed_norm': levenshtein(ref, pred) / max(1, len(ref))}
    for name, n in [('rouge1', overlap), ('rougeL', common)]:
        precision, recall = n / len(b) if b else 0.0, n / len(a) if a else 0.0
        result[name + '_precision'] = precision
        result[name + '_recall'] = recall
        result[name + '_f1'] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return result


def metric_selfcheck(original):
    rng = random.Random(73)
    for _ in range(600):
        a = ''.join(rng.choices("ab c\nżé0", k=rng.randrange(45)))
        b = ''.join(rng.choices("ab c\nżé0", k=rng.randrange(45)))
        assert levenshtein(a, b) == original.levenshtein_distance(a, b)
        aa, bb = a.split(), b.split()
        assert lcs(aa, bb) == original.lcs_len(aa, bb)
    for a, b in [('', ''), ('', 'a'), ('abc', ''), ('kitten', 'sitting'), ('x'*200, 'y'*200)]:
        assert levenshtein(a, b) == original.levenshtein_distance(a, b)


def compare_numeric(old, new, prefix=''):
    differences = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old.keys() | new.keys():
            if k not in old or k not in new:
                differences.append({'path': prefix + '/' + k, 'kind': 'missing'})
            else:
                differences += compare_numeric(old[k], new[k], prefix + '/' + k)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            differences.append({'path': prefix, 'kind': 'length'})
        else:
            for i, (a, b) in enumerate(zip(old, new)):
                differences += compare_numeric(a, b, prefix + '/' + str(i))
    elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if old != new:
            differences.append({'path': prefix, 'old': old, 'new': new, 'abs_diff': abs(old-new)})
    elif old != new:
        differences.append({'path': prefix, 'kind': 'value', 'old': old, 'new': new})
    return differences


def signed_rank(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if not len(d):
        return {'n_nonzero': 0, 'p_raw': 1.0, 'rank_biserial': 0.0, 'z': 0.0}
    r = spstats.rankdata(abs(d), method='average')
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    sd = math.sqrt(float(np.dot(r, r))/4)
    delta = min(wp, wm) - (wp+wm)/2
    z = (delta + .5 if delta < 0 else 0.0) / sd
    # erfc keeps extreme tails that subtraction from erf rounded to zero.
    p = math.erfc(abs(z) / math.sqrt(2))
    scipy_p = float(spstats.wilcoxon(d, zero_method='wilcox', correction=True, alternative='two-sided', mode='approx').pvalue)
    assert abs(p - scipy_p) < 1e-12
    return {'n_nonzero': len(d), 'p_raw': p, 'rank_biserial': float((wp-wm)/(wp+wm)), 'z': float(z)}


def holm(pairs):
    running = 0.0
    for rank, i in enumerate(sorted(range(len(pairs)), key=lambda i: pairs[i]['p_raw'])):
        running = max(running, (len(pairs)-rank) * pairs[i]['p_raw'])
        pairs[i]['p_holm'] = min(1.0, running)
        pairs[i]['significant_0_05'] = pairs[i]['p_holm'] < .05


def ci(x, rng):
    x = np.asarray(x)
    draws = x[rng.integers(len(x), size=(BOOT, len(x)))].mean(axis=1)
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def main():
    assert os.environ.get('PYTHONHASHSEED') == '0', 'Invoke with PYTHONHASHSEED=0 to replay the documented overhead recipe.'
    orig = module('run_validation')
    stat = module('analyze_stats')
    extra = module('analyze_extras')
    robustness = module('robustness_eval')
    prompts = module('generate_prompts')
    metric_selfcheck(orig)
    sources = [SRC / f'{n}.py' for n in ['run_validation','analyze_stats','analyze_extras','robustness_eval','generate_prompts','defense_eval','trace_baselines']]
    primary = {}
    metrics_out = []
    differences = defaultdict(list)
    max_diff = defaultdict(float)
    main_count = 0
    for label, _, run in stat.RUNS:
        path = stat.RESULTS / run / 'samples.jsonl'
        sources.append(path)
        rows = readrows(path)
        assert len(rows) == 300 and len({r['idx'] for r in rows}) == 300
        primary[label] = {r['idx']: r for r in rows}
        for row in rows:
            matched = [s for s in prompts.TOPICS[row['topic']]['issues'] if s.lower() in row['prompt'].lower()]
            assert len(matched) == 1, (row['idx'], matched)
            scenario = row['topic'] + '::' + matched[0]
            result = metric(row['response_text'], row['pred_full_text'])
            first = metric(row['response_first_sentence'], row['pred_first_sentence'])
            compare = {k: v for k, v in result.items() if k != 'phi'}
            compare.update({'first_ed_norm': first['ed_norm'], 'first_rouge1_precision': first['rouge1_precision'], 'first_rougeL_precision': first['rougeL_precision']})
            for key, value in compare.items():
                delta = abs(value - row[key]); max_diff['primary/' + key] = max(max_diff['primary/'+key], delta)
                if delta > 1e-12:
                    differences['primary'].append({'label': label, 'idx': row['idx'], 'metric': key, 'stored': row[key], 'recomputed': value})
                row[key] = value
            row['_phi'] = result['phi'];row['_scenario'] = scenario
            metrics_out.append({'label':label,'idx':row['idx'],'run_id':row['run_id'],'scenario':scenario,**result})
            main_count += 1
        print('Recomputed full/first text metrics:',label,flush=True)
    align = next(iter(primary.values()))
    for label, rows in primary.items():
        assert all((r['topic'], r['prompt']) == (align[i]['topic'], align[i]['prompt']) for i, r in rows.items())
        assert set(Counter(r['_scenario'] for r in rows.values()).values()) == {5}
    (OUT/'primary_rescored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in metrics_out))
    mapping = {run:label for label,_,run in stat.RUNS}
    def corrected_load(run):
        return {i:dict(idx=i,topic=r['topic'],phi=r['_phi'],ed_norm=r['ed_norm'],r1=r['rouge1_precision'],rl=r['rougeL_precision'],response_complete=r['response_complete'],finish_reason=r['finish_reason'],token_count=r['token_count']) for i,r in primary[mapping[run]].items()}
    stat.load_run = corrected_load;stat.OUT = RERUN
    stat.main()
    # Re-run supplemental analyses under the documented process-hash seed.
    extra.OUT = RERUN
    shutil.copy2(OLD/'defense_samples.jsonl', RERUN/'defense_samples.jsonl')
    extra.main()

    # Independent signed-rank calculation and scenario-level sensitivity.
    labels = [r[0] for r in stat.RUNS];scenarios=sorted({r['_scenario'] for r in align.values()})
    grouped = {lab:np.array([np.mean([r['_phi'] for r in primary[lab].values() if r['_scenario']==s]) for s in scenarios]) for lab in labels}
    arrays = {lab:np.array([primary[lab][i]['_phi'] for i in sorted(align)]) for lab in labels}
    rng=np.random.default_rng(SEED);cluster_stats=[]
    for lab in labels:
        cluster_stats.append({'label':lab,'n_scenarios':60,'n_prompts':300,'phi_mean':float(grouped[lab].mean()),'phi_ci_scenario_bootstrap':ci(grouped[lab],rng)})
    primary_tests=[];cluster_tests=[]
    for a,b in itertools.combinations(labels,2):
        primary_tests.append(dict(a=a,b=b,mean_diff_phi=float((arrays[a]-arrays[b]).mean()),**signed_rank(arrays[a],arrays[b])))
        cluster_tests.append(dict(a=a,b=b,mean_diff_phi=float((grouped[a]-grouped[b]).mean()),diff_ci_scenario_bootstrap=ci(grouped[a]-grouped[b],rng),**signed_rank(grouped[a],grouped[b])))
    holm(primary_tests);holm(cluster_tests)
    save('independent_primary_pairwise.json',primary_tests)
    save('scenario_cluster_sensitivity.json',{'method':'Exploratory sensitivity: average five prompt variants within each of 60 scenarios; resample scenarios for mean CIs; paired signed-rank test on scenario averages, normal approximation with continuity correction; Holm over 21 pairs. This is a different test unit and does not repair measurement errors.','models':cluster_stats,'pairs':cluster_tests})

    # Auxiliary outputs: rescore raw predictions against their actual source reference.
    pilot={}
    for label,_,run in extra.DEFENSE_RUNS:
        path=extra.RESULTS/run/'samples.jsonl';sources.append(path)
        pilot[label]={r['idx']:r for r in readrows(path)}
    sets={}
    for name,keymap,use_primary in [
        ('trace_baseline_samples',{'phi':'phi'},False),
        ('defense_samples',{'phi':'phi','ed_norm':'ed_norm','r1':'rouge1_precision'},False),
        ('robustness_samples',{'phi':'phi','ed_norm':'ed_norm','r1_precision':'rouge1_precision'},True)]:
        path=OLD/(name+'.jsonl');sources.append(path);rows=readrows(path)
        for row in rows:
            source=(primary if use_primary else pilot)[row['label']][row['idx']]
            result=metric(source['response_text'],row['pred'])
            for stored,new in keymap.items():
                delta=abs(row[stored]-result[new]);max_diff[name+'/'+stored]=max(max_diff[name+'/'+stored],delta)
                if delta>1e-12: differences[name].append({'label':row['label'],'idx':row['idx'],'condition':row.get('condition',row.get('defense')),'metric':stored,'stored':row[stored],'recomputed':result[new]})
                row[stored]=result[new]
        sets[name]=rows
        (OUT/(name+'.rescored.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        print('Rescored',name,len(rows),flush=True)
    # Control CIs with fixed percentile unpacking, preserving recorded order.
    control=[];rng=np.random.default_rng(SEED)
    for label,pretty,_ in extra.DEFENSE_RUNS:
        real=np.array([r['phi'] for r in sets['trace_baseline_samples'] if r['label']==label and r['condition']=='real'])
        for condition in ['real','shuffled','constant','cross_prompt']:
            arr=np.array([r['phi'] for r in sets['trace_baseline_samples'] if r['label']==label and r['condition']==condition])
            control.append({'label':label,'model':pretty,'condition':condition,'n':len(arr),'phi_mean':float(arr.mean()),'phi_ci':ci(arr,rng),'delta_vs_real':float(arr.mean()-real.mean())})
    (RERUN/'trace_baselines.json').write_text(json.dumps(control,indent=2)+'\n')
    # Defense summaries: independent transforms; seed-uncertain random overhead is distinguished below.
    defense=[];defense_tests=[];rng=np.random.default_rng(SEED)
    for label,pretty,_ in extra.DEFENSE_RUNS:
        base={r['idx']:r for r in sets['defense_samples'] if r['label']==label and r['defense']=='baseline'}
        for condition in ['baseline','bucket_8','pad_32','batch_2','batch_4','rand_pad_8']:
            rows=[r for r in sets['defense_samples'] if r['label']==label and r['defense']==condition]
            assert len(rows)==30 and set(r['idx'] for r in rows)==set(base)
            phi=np.array([r['phi'] for r in rows]);ed=np.array([r['ed_norm'] for r in rows]);r1=np.array([r['r1'] for r in rows])
            overs=[];frames=[];expected_random=[];random_lower=[];random_upper=[]
            for row in rows:
                toks=extra.token_lengths_for(pilot[label][row['idx']]['run_id']);assert toks
                seed=(SEED+row['idx']*131+hash(condition)%9973)&0x7fffffff
                if condition=='baseline': transformed=toks
                elif condition=='bucket_8':transformed=[8*((x+7)//8) for x in toks]
                elif condition=='pad_32':assert max(toks)<=32;transformed=[32]*len(toks)
                elif condition.startswith('batch_'):k=int(condition[-1]);transformed=[sum(toks[i:i+k]) for i in range(0,len(toks),k)]
                else:transformed=(np.array(toks)+np.random.default_rng(seed).integers(0,9,len(toks))).tolist()
                assert len(transformed)==row['n_frames']
                bb=sum(toks);overs.append(100*(sum(transformed)-bb)/bb);frames.append(len(transformed));expected_random.append(400*len(toks)/bb);random_lower.append(0.0);random_upper.append(800*len(toks)/bb)
            lr,lo,hi=stat.wilson_ci(int(sum(phi>.5)),len(phi))
            d={'label':label,'model':pretty,'defense':condition,'n':len(phi),'phi_mean':float(phi.mean()),'phi_ci':ci(phi,rng),'ed_norm_mean':float(ed.mean()),'r1_mean':float(r1.mean()),'lr_at_0_5_pct':100*lr,'lr_ci':[100*lo,100*hi],'asr':100*lr,'mean_frames':float(np.mean(frames)),'byte_overhead_pct_mean':float(np.mean(overs)),'leakage_reduction_pct':float(100*(1-phi.mean()/np.mean([r['phi'] for r in base.values()])))}
            defense.append(d)
            if condition=='rand_pad_8':
                save(f'{label}_random_padding_expected_overhead.json',{'scope':'Distribution-defined expected payload overhead independent of unavailable historical padding realizations. This is an expectation, not the empirical overhead of an archived inference run.','mean_expected_overhead_pct':float(np.mean(expected_random)),'possible_mean_overhead_range_pct':[0.0,float(np.mean(random_upper))],'replayed_recipe_mean_overhead_pct':d['byte_overhead_pct_mean'],'recipe_pythonhashseed':0})
            if condition!='baseline':
                basearr=[base[r['idx']]['phi'] for r in rows]
                defense_tests.append(dict(label=label,defense=condition,mean_difference=float(np.mean(phi-np.array(basearr))),**signed_rank(phi,basearr)))
    holm(defense_tests)
    (RERUN/'defense_eval.json').write_text(json.dumps(defense,indent=2)+'\n')
    save('independent_defense_pairwise.json',defense_tests)
    config=json.loads((OLD/'robustness_samples_config.json').read_text());sources.append(OLD/'robustness_samples_config.json')
    result=robustness.aggregate(OUT/'robustness_samples.rescored.jsonl',config,config['protocol_id'],BOOT)
    assert result['complete']
    # Independently check all source cross-product keys and stable seeds; duplicates forbidden.
    for label in config['run_labels']:
        selected=[r for r in sets['robustness_samples'] if r['label']==label]
        indexes={r['idx'] for r in selected}
        assert len({primary[label][i]['_scenario'] for i in indexes})==30
    keys=[(r['label'],r['idx'],r['defense'],r['attacker'],r['seed_index']) for r in sets['robustness_samples']]
    assert len(keys)==len(set(keys))==3300
    assert all(r['decoder_seed']==robustness.common_decoder_seed(r['seed_index'],r['idx']) for r in sets['robustness_samples'])
    (RERUN/'robustness_eval.json').write_text(json.dumps(result,indent=2)+'\n')
    comparisons={}
    for name in ['model_stats.json','pairwise_tests.json','covariate_regression.json','topic_matrix.json','defense_overhead.json','trace_baselines.json','defense_eval.json','robustness_eval.json']:
        prior=json.loads((OLD/name).read_text());now=json.loads((RERUN/name).read_text());diff=compare_numeric(prior,now)
        comparisons[name]={'exact_json_value_match':not diff,'max_numeric_abs_diff':max([x.get('abs_diff',0) for x in diff],default=0),'difference_count':len(diff),'differences':diff}
    save('aggregate_comparisons.json',comparisons)
    save('metric_checks.json',{'primary_rows':main_count,'first_response_rows_checked':main_count,'auxiliary_rows':{k:len(v) for k,v in sets.items()},'independent_algorithm_selfchecks':605,'max_abs_difference_from_stored_fields':dict(max_diff),'mismatches_above_1e_12':dict(differences),'definition':'TF cosine, raw-character reference-normalized Levenshtein, regex word ROUGE1/L. No embedding-backend historical phi comparison; primary phi is recomputed consistently.'})
    save('summary.json',{'verification_status':'VERIFIED calculations conditional on archived text; collection/reference validity unresolved','python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pythonhashseed':os.environ['PYTHONHASHSEED'],'primary_pairs_significant_21':sum(x['significant_0_05'] for x in primary_tests),'scenario_pairs_significant_21':sum(x['significant_0_05'] for x in cluster_tests),'defense_pairs_significant_10':sum(x['significant_0_05'] for x in defense_tests),'metric_mismatch_count':sum(len(x) for x in differences.values()),'source_bugs':['trace_baselines.py percentile interval unpacking attempts float(array)','Signed-rank p-value subtraction loses extreme tails; independent erfc/SciPy values retained'],'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]})
    print('COMPLETE',json.loads((OUT/'summary.json').read_text())['scenario_pairs_significant_21'], 'scenario-significant pairs',flush=True)


if __name__=='__main__':
    main()
