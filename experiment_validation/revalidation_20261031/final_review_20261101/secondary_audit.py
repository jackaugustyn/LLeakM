"""Independent arithmetic audit of the completed 3900-record secondary rerun.

Read-only for all experiment artifacts. Outputs only secondary_audit*. No model
loading or inference; recomputes all text metrics, costs, intervals and contrasts.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
RUN = BASE/'secondary/cuda_full'
WORD = re.compile(r"[A-Za-z0-9']+")
SEEDS = list(range(20260706,20260711))
BOOT = 10000


def readrows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seed(*parts):
    return int.from_bytes(hashlib.blake2s('\x1f'.join(map(str,parts)).encode(),digest_size=4).digest(),'big')&0x7fffffff


def levenshtein(a,b):
    if len(a)>len(b):a,b=b,a
    if not a:return len(b)
    masks=defaultdict(int)
    for i,c in enumerate(a):masks[c]|=1<<i
    vp,vn,score,last=~0,0,len(a),1<<(len(a)-1)
    for c in b:
        x=masks[c]|vn;d0=(((x&vp)+vp)^vp)|x
        hp=vn|~(d0|vp);hn=d0&vp
        score+=bool(hp&last)-bool(hn&last)
        hp=(hp<<1)|1;vn=d0&hp;vp=(hn<<1)|~(d0|hp)
    return score


def lcs(a,b):
    masks=defaultdict(int)
    for i,c in enumerate(b):masks[c]|=1<<i
    s=0
    for c in a:
        u=masks[c]|s;s=u&~(u-((s<<1)|1))
    return s.bit_count()


def metrics(ref,pred):
    a,b=WORD.findall(ref.lower()),WORD.findall(pred.lower())
    ca,cb=Counter(a),Counter(b)
    denom=math.sqrt(sum(x*x for x in ca.values()))*math.sqrt(sum(x*x for x in cb.values()))
    phi=sum(ca[x]*cb[x] for x in ca)/denom if denom else 0.
    result={'phi':phi,'ed_norm':levenshtein(ref,pred)/max(1,len(ref))}
    for name,n in [('rouge1',sum((ca&cb).values())),('rougeL',lcs(a,b))]:
        p=n/len(b) if b else 0.;r=n/len(a) if a else 0.
        result[name+'_precision']=p;result[name+'_recall']=r
        result[name+'_f1']=2*p*r/(p+r) if p+r else 0.
    return result


def boot_ci(matrix,seed_value,crossed):
    rng=np.random.default_rng(seed_value)
    a=np.asarray(matrix,dtype=float)
    if not crossed:
        a=a.ravel();means=a[rng.integers(0,len(a),size=(BOOT,len(a)))].mean(axis=1)
    else:
        # Resample the prompt and seed axes independently, retaining their
        # crossing. Interleaved random draws reproduce the documented recipe.
        means=[]
        for _ in range(BOOT):
            pi=rng.integers(a.shape[0],size=a.shape[0])
            si=rng.integers(a.shape[1],size=a.shape[1])
            means.append(a[pi][:,si].mean())
    return np.quantile(means,[.025,.975]).tolist()


def wilson(k,n):
    z=1.959963984540054;p=k/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [100*(center-half),100*(center+half)]


def signed_rank(a,b):
    d=np.asarray(a).ravel()-np.asarray(b).ravel();d=d[d!=0]
    if not len(d):return {'p_value':1.,'rank_biserial':0.,'n':0}
    ranks=stats.rankdata(abs(d));total=ranks.sum()
    p=float(stats.wilcoxon(d,correction=True,zero_method='wilcox',mode='approx').pvalue)
    return {'p_value':p,'rank_biserial':float((ranks[d>0].sum()-ranks[d<0].sum())/total),'n':len(d)}


def main():
    data=readrows(RUN/'samples.jsonl');inventory=readrows(RUN/'input_inventory.jsonl')
    summary=json.loads((RUN/'summary.json').read_text());protocol=json.loads((RUN/'protocol.json').read_text())
    assert summary['complete'] and len(data)==len(inventory)==summary['n']==3900
    assert digest(protocol)==summary['protocol_id']
    assert sha(RUN/'input_inventory.jsonl')==protocol['input_inventory_sha256']
    assert sha(RUN/'calibration.json')==protocol['calibration_sha256']
    source_hashes={p:sha(ROOT/p) for p in protocol['sources']}
    assert source_hashes==protocol['sources']
    taskmap={r['task_id']:r for r in inventory};assert len(taskmap)==3900
    assert Counter(r['suite'] for r in data)=={'trace_control':240,'defense_pilot':360,'map':3300}
    assert len({r['task_id'] for r in data})==3900
    maxdiff=defaultdict(float);grouped=defaultdict(list);rechecked=[]
    logs={};all_phi_ties=0
    for row in data:
        item=taskmap[row['task_id']]
        assert digest({k:v for k,v in row.items() if k!='record_sha256'})==row['record_sha256']
        assert all(row[k]==v for k,v in item.items())
        assert digest({k:v for k,v in item.items() if k!='input_sha256'})==item['input_sha256']
        assert hashlib.sha256(row['reference'].encode()).hexdigest()==row['reference_sha256']
        for name in ['clean_lengths','observed_lengths','attacker_input_lengths']:
            assert digest(row[name])==row[name+'_sha256']
        if row['run_id'] not in logs:
            path=ROOT/'logs'/(row['run_id']+'.json')
            assert sha(path)==row['generation_log_sha256']
            logs[row['run_id']]=json.loads(path.read_text())
        log=logs[row['run_id']]
        assert log['response_text']==row['reference']==''.join(s['token_text'] for s in log['steps'])
        assert [len(s['token_text'].encode()) for s in log['steps']]==row['clean_lengths']
        new=metrics(row['reference'],row['prediction'])
        for name,value in new.items():
            difference=abs(value-row['metrics'][name]);maxdiff[name]=max(maxdiff[name],difference)
            assert difference<1e-12,(row['task_id'],name,difference)
        overhead=100*(sum(row['observed_lengths'])-sum(row['clean_lengths']))/sum(row['clean_lengths'])
        assert abs(overhead-row['observed_payload_overhead_pct'])<1e-12
        assert row['n_observed_units']==len(row['observed_lengths'])
        assert row['n_attacker_input_units']==len(row['attacker_input_lengths'])
        actual=row['actual_encoder_inputs']
        selected=[c['selected_text'].strip() for c in actual if c['selected_text'].strip()]
        assert re.sub(r'\s+([,.;:!?])',r'\1',' '.join(selected))==row['prediction']
        assert row['attempted_segments']==len(actual)
        for call in actual:
            assert len(call['full_token_ids'])==call['full_token_count']
            assert len(call['retained_token_ids'])<=80
            assert call['truncated']==(call['full_token_count']>len(call['retained_token_ids']))
        all_phi_ties+=row['metrics']['phi']==.5
        key=tuple(row[k] for k in ['suite','label','condition','attacker'])
        grouped[key].append(row)
        rechecked.append({'task_id':row['task_id'],**new})
    cells={tuple(c[k] for k in ['suite','label','condition','attacker']):c for c in summary['cells']}
    assert set(cells)==set(grouped) and len(cells)==42
    facts=[];matrices={};ci_max=0.;means_max=0.
    for key,rr in grouped.items():
        cell=cells[key];crossed=key[0]=='map';rr=sorted(rr,key=lambda r:(r['idx'],r['seed_index']))
        indices=sorted({r['idx'] for r in rr});seeds=SEEDS if crossed else [20260706]
        assert len(indices)==30
        assert {(r['idx'],r['seed_index']) for r in rr}==set(itertools.product(indices,seeds))
        values={(r['idx'],r['seed_index']):r['metrics']['phi'] for r in rr}
        matrix=np.array([[values[i,s] for s in seeds] for i in indices]);matrices[key]=matrix
        mean_metrics={m:float(np.mean([r['metrics'][m] for r in rr])) for m in rr[0]['metrics']}
        for metric,value in mean_metrics.items():
            diff=abs(value-cell['metric_means'][metric]);means_max=max(means_max,diff);assert diff<1e-12
        ci=boot_ci(matrix,seed('bootstrap',*key),crossed)
        storedci=cell['phi_ci_crossed' if crossed else 'phi_ci_prompt']
        cidiff=max(abs(a-b) for a,b in zip(ci,storedci));ci_max=max(ci_max,cidiff);assert cidiff<1e-12
        cost=float(np.mean([100*(sum(r['observed_lengths'])/sum(r['clean_lengths'])-1) for r in rr]))
        assert abs(cost-cell['observed_payload_overhead_pct_mean'])<1e-10
        count=sum(r['metrics']['phi']>.5 for r in rr);assert abs(100*count/len(rr)-cell['lr_at_0_5_pct'])<1e-12
        expected_rand=float(np.mean([100*4*len(r['clean_lengths'])/sum(r['clean_lengths']) for r in rr])) if key[2]=='rand_pad_8' else None
        coverage=[]
        for i in indices:
            one=[r for r in rr if r['idx']==i]
            # First inputs are seed-independent, but later context is not.
            firsts=[r['actual_encoder_inputs'][0] for r in one]
            assert all(f['text']==firsts[0]['text'] for f in firsts)
            coverage.append(firsts[0]['truncated'])
        fact={k:v for k,v in zip(['suite','label','condition','attacker'],key)}
        fact.update(n=len(rr),n_prompts=30,n_seeds=len(seeds),metric_means=mean_metrics,phi_ci=ci,
                    observed_payload_overhead_pct_mean=cost,expected_random_padding_overhead_pct_mean=expected_rand,
                    observed_units_mean=cell['observed_units_mean'],lr_at_0_5_count=count,lr_at_0_5_pct=100*count/len(rr),
                    lr_wilson_ci_pct=wilson(count,len(rr)) if not crossed else None,
                    encoder_calls=sum(len(r['actual_encoder_inputs']) for r in rr),
                    encoder_calls_truncated=sum(c['truncated'] for r in rr for c in r['actual_encoder_inputs']),
                    unique_first_inputs_truncated=sum(coverage),
                    stopped_before_requested_count=sum(r['stopped_before_requested_segments'] for r in rr),
                    empty_first_prediction_count=sum(r['empty_first_prediction'] for r in rr))
        if crossed:
            sd=float(np.std(matrix.mean(axis=0),ddof=1));assert abs(sd-cell['between_seed_sd'])<1e-12
            fact.update(between_seed_sd=sd,seed_specific_means=matrix.mean(axis=0).tolist(),
                        individual_phi_range=[float(matrix.min()),float(matrix.max())])
        facts.append(fact)
    contrasts=[];bycontrast={tuple(c[k] for k in ['suite','label','condition','attacker']):c for c in summary['paired_contrasts']}
    for key,matrix in matrices.items():
        base=(key[0],key[1],'real' if key[0]=='trace_control' else 'baseline','frozen')
        if base==key:continue
        old=bycontrast[key];diff=matrix-matrices[base]
        value=float(diff.mean());reduction=100*(1-float(matrix.mean())/float(matrices[base].mean()))
        assert abs(value-old['mean_phi_difference_vs_baseline'])<1e-12
        assert abs(reduction-old['score_reduction_pct'])<1e-10
        crossed=key[0]=='map';ci=boot_ci(diff,seed('contrast',*key),crossed)
        cidiff=max(abs(a-b) for a,b in zip(ci,old['ci_crossed' if crossed else 'ci_prompt']));ci_max=max(ci_max,cidiff);assert cidiff<1e-12
        c={k:v for k,v in zip(['suite','label','condition','attacker'],key)}
        c.update(mean_phi_difference_vs_baseline=value,score_reduction_pct=reduction,ci=ci)
        if not crossed:
            sr=signed_rank(matrix,matrices[base]);c.update(sr)
            assert abs(sr['p_value']-old['p_value'])<1e-12 and abs(sr['rank_biserial']-old['rank_biserial'])<1e-12
        contrasts.append(c)
    for suite in ['trace_control','defense_pilot']:
        family=[c for c in contrasts if c['suite']==suite];running=0.
        for rank,c in enumerate(sorted(family,key=lambda r:r['p_value'])):
            running=max(running,(len(family)-rank)*c['p_value']);c['p_holm']=min(1.,running)
            c['significant_holm_0_05']=c['p_holm']<.05
            key=tuple(c[k] for k in ['suite','label','condition','attacker'])
            assert abs(c['p_holm']-bycontrast[key]['p_holm'])<1e-12
    archived=readrows(ROOT/'experiment_validation/analysis/robustness_samples.jsonl')
    oldmap={(r['label'],r['idx'],r['defense'],r['attacker'],r['seed_index']):r for r in archived}
    maps=[r for r in data if r['suite']=='map'];matches=0;seedmatches=0
    for r in maps:
        old=oldmap[r['label'],r['idx'],r['condition'],r['attacker'],r['seed_index']]
        matches+=old['pred']==r['prediction'];seedmatches+=all(old[k]==r[k] for k in ['decoder_seed','trace_seed'])
        assert r['decoder_seed']==seed('decoder',r['seed_index'],r['idx'])
    mapfacts=[c for c in facts if c['suite']=='map'];defcontrasts=[c for c in contrasts if c['suite']=='defense_pilot']
    figure=[]
    for condition in ['bucket_8','pad_32','batch_2','batch_4','rand_pad_8']:
        ff=[c for c in facts if c['suite']=='defense_pilot' and c['condition']==condition]
        cc=[c for c in defcontrasts if c['condition']==condition]
        figure.append({'condition':condition,'payload_overhead_pct':float(np.mean([c['observed_payload_overhead_pct_mean'] for c in ff])),
                       'mean_model_score_reduction_pct':float(np.mean([c['score_reduction_pct'] for c in cc]))})
    report={'audited_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','protocol_id':summary['protocol_id'],
            'source_files':{str(p.relative_to(ROOT)):sha(p) for p in [RUN/'samples.jsonl',RUN/'summary.json',RUN/'input_inventory.jsonl',RUN/'protocol.json',Path(__file__)]},
            'source_hashes_verified':len(source_hashes),'unique_generation_logs_bound_to_scored_references':len(logs),
            'counts':dict(Counter(r['suite'] for r in data)),'all_3900_record_digests_valid':True,
            'all_3900_input_bindings_valid':True,'metric_max_abs_difference':dict(maxdiff),'cell_metric_mean_max_abs_difference':means_max,
            'bootstrap_interval_max_abs_difference':ci_max,'exactly_phi_0_5_records':all_phi_ties,
            'cells':facts,'paired_contrasts':contrasts,'defense_figure_equal_model_average':figure,
            'ranges':{'defense_score_reduction_pct':[min(c['score_reduction_pct'] for c in defcontrasts),max(c['score_reduction_pct'] for c in defcontrasts)],
                      'map_cell_phi_mean':[min(c['metric_means']['phi'] for c in mapfacts),max(c['metric_means']['phi'] for c in mapfacts)],
                      'map_between_seed_sd':[min(c['between_seed_sd'] for c in mapfacts),max(c['between_seed_sd'] for c in mapfacts)],
                      'map_seed_specific_cell_means':[min(min(c['seed_specific_means']) for c in mapfacts),max(max(c['seed_specific_means']) for c in mapfacts)]},
            'significant_control_contrasts':sum(c.get('significant_holm_0_05',False) for c in contrasts if c['suite']=='trace_control'),
            'significant_defense_contrasts':sum(c.get('significant_holm_0_05',False) for c in defcontrasts),
            'map_historical_reproduction':{'exact_prediction_matches':matches,'matched_records':len(maps),'seed_pair_matches':seedmatches,
                'bootstrap_note':'Means, reductions and seed SDs reproduce; new crossed-bootstrap endpoints differ because the rerun uses suite/condition-specific bootstrap seeds.'},
            'interpretation_limits':['Controls are partial ablations preserving some response-dependent information.',
                'Pilot condition-specific decoder seeds are new explicit BLAKE2s seeds; this does not recover unknown historical pilot RNG states.',
                'MAP has only first-segment reconstruction with a full-response reference; transformed inputs have unequal encoder coverage.',
                'Prompt and seed repeats are crossed, not independent new prompts; calibration/test are exact-prompt-disjoint but share scenarios.',
                'Padding costs are payload-only; recorded random-padding realizations are now reproducible and differ from expected cost.',
                'No transformation contrast identifies information-theoretic secrecy or semantic disclosure.']}
    (HERE/'secondary_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    (HERE/'secondary_audit_metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rechecked))
    text=['# Końcowa kontrola dodatkowych 3900 rekonstrukcji','',
          'Status: **PASS**. Zweryfikowano 240 kontroli, 360 prób obron i 3300 prób MAP. Nie wykonywano inferencji ani nie zmieniano poświadczonych artefaktów.',
          '',f"Ponownie obliczono wszystkie metryki tekstowe, narzuty, 42 średnie komórkowe, 78 przedziałów bootstrapu i 36 kontrastów. Największa różnica metryk: {max(maxdiff.values()):.3g}; przedziałów: {ci_max:.3g}.",
          '',f"Kontrole: {report['significant_control_contrasts']}/6 istotnych porównań po Holm. Obrony: {report['significant_defense_contrasts']}/10. Redukcje pilotowe: {report['ranges']['defense_score_reduction_pct'][0]:.4f}–{report['ranges']['defense_score_reduction_pct'][1]:.4f}%.",
          '',f"Wszystkie {matches}/3300 predykcji MAP odtworzono dokładnie względem archiwum. Odmienione przedziały wynikają z nowego jawnego przepisu bootstrapu, nie zmiany predykcji.",
          '', 'Dokładne liczby tabel, zakresy, dane wykresu i pokrycie enkodera: `secondary_audit.json`. Średnie koszty losowego paddingu są zarejestrowanymi nowymi realizacjami; nie należy mieszać ich z wcześniej obliczonymi wartościami oczekiwanymi.', '']
    (HERE/'secondary_audit.md').write_text('\n'.join(text))
    print(json.dumps({k:report[k] for k in ['status','counts','metric_max_abs_difference','bootstrap_interval_max_abs_difference','significant_control_contrasts','significant_defense_contrasts','ranges','map_historical_reproduction']},indent=2))


if __name__=='__main__':main()
