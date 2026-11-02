"""Final read-only arithmetic audit and article fact ledger, no inference."""
from __future__ import annotations
from array import array
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy import stats

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]
WORD=re.compile(r"[A-Za-z0-9']+")
COHORTS=['cuda_full','fresh_tinyllama_cuda','fresh_phi_cuda']


def readrows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def save(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def cpp_metrics():
    source=HERE/'independent_dp.cpp'; library=HERE/'independent_dp.so'
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    dll=ctypes.CDLL(str(library))
    for name in ['edit_distance','longest_common_subsequence']:
        f=getattr(dll,name)
        f.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint32),ctypes.c_size_t]
        f.restype=ctypes.c_uint32
    return dll


def call_dp(f,a,b):
    aa=array('I',a);bb=array('I',b)
    av=(ctypes.c_uint32*len(aa)).from_buffer(aa)
    bv=(ctypes.c_uint32*len(bb)).from_buffer(bb)
    return int(f(av,len(aa),bv,len(bb)))


def metric(dll,ref,pred):
    a,b=WORD.findall(ref.lower()),WORD.findall(pred.lower())
    ca,cb=Counter(a),Counter(b)
    den=math.sqrt(sum(x*x for x in ca.values())*sum(x*x for x in cb.values()))
    result={'phi':sum(v*cb[k] for k,v in ca.items())/den if den else 0.,
            'ed_norm':call_dp(dll.edit_distance,map(ord,ref),map(ord,pred))/max(1,len(ref))}
    vocabulary={w:i for i,w in enumerate(set(a+b))}
    common=call_dp(dll.longest_common_subsequence,(vocabulary[x] for x in a),(vocabulary[x] for x in b))
    for name,n in [('rouge1',sum(min(v,cb[k]) for k,v in ca.items())),('rougeL',common)]:
        p=n/len(b) if b else 0.;r=n/len(a) if a else 0.
        result[name+'_precision']=p;result[name+'_recall']=r
        result[name+'_f1']=2*p*r/(p+r) if p+r else 0.
    return result


def first_sentence(text):
    t=text.strip()
    m=re.search(r'(.+?[.!?])(?:\s|$)',t,flags=re.S)
    return m.group(1).strip() if m else (t.splitlines()[0].strip() if t else '')


def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den
    d=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,c-d),min(1.,c+d)]


def boot(x,rng,n=10000):
    a=np.asarray(x);means=a[rng.integers(len(a),size=(n,len(a)))].mean(axis=1)
    return np.percentile(means,[2.5,97.5]).tolist()


def signed_rank(a,b):
    d=np.asarray(a)-np.asarray(b);d=d[d!=0]
    if not len(d):return {'p_raw':1.,'rank_biserial':0.}
    p=float(stats.wilcoxon(d,zero_method='wilcox',correction=True,alternative='two-sided',mode='approx').pvalue)
    ranks=stats.rankdata(abs(d),method='average')
    return {'p_raw':p,'rank_biserial':float((ranks[d>0].sum()-ranks[d<0].sum())/ranks.sum())}


def regression(raw):
    rows=[r for r in raw if r['label']!='phi_1_5'];labels=list(dict.fromkeys(r['label'] for r in rows))
    y=np.asarray([r['phi'] for r in rows]);dummies=np.asarray([[1.]+[float(r['label']==lab) for lab in labels[1:]] for r in rows])
    cov=np.asarray([[len(r['reference'])/100.,np.mean(r['token_lengths']),float(len(r['token_lengths'])>=512)] for r in rows])
    out={'n':len(rows),'labels':labels,'estimand':'Descriptive OLS including two malformed-reference diagnostic series; no causal attribution','source_features':'Intercept plus model indicators (first label omitted); reference Unicode characters/100, mean logged token bytes, token count>=512'}
    total=np.sum((y-y.mean())**2)
    for name,x in [('model_only',dummies),('covariates_only',np.column_stack([np.ones(len(rows)),cov])),('full',np.column_stack([dummies,cov]))]:
        beta=np.linalg.lstsq(x,y,rcond=None)[0]
        out['r2_'+name]=float(1-np.sum((y-x@beta)**2)/total)
    out['delta_r2_full_minus_model']=out['r2_full']-out['r2_model_only']
    return out


def main():
    dll=cpp_metrics()
    assert call_dp(dll.edit_distance,map(ord,'kitten'),map(ord,'sitting'))==3
    assert call_dp(dll.edit_distance,[],map(ord,'ż🙂'))==2
    ledger={'audited_utc':datetime.now(timezone.utc).isoformat(),'cohorts':{},'bootstrap_seed_verified':20260706,
            'bootstrap_draws_verified':10000,'LR_definition':'phi > 0.5; exact ties checked explicitly',
            'Wilson_z':1.959963984540054,'metric_audit':'Text metrics independently recalculated by textbook C++ dynamic programs (Unicode ED and word LCS) and Python word counts; no model inference.',
            'limits':['Historical TinyLlama/Phi references remain diagnostics.','Fresh cohorts have separately pinned victim generations.','Corrected selector uses fixed traces, references and per-record seeds; end-to-end generation paths can diverge after changed context.','Lexical metrics are not calibrated disclosure rates.']}
    maxdiff=Counter();checked=0
    for ci,name in enumerate(COHORTS):
        run=BASE/'inference'/name;vdir=BASE/'inference_validation'/name
        raw=readrows(run/'samples.jsonl');rescored=readrows(vdir/'rescored.jsonl')
        stats_file=json.loads((vdir/'statistics.json').read_text());check=json.loads((vdir/'verification.json').read_text())
        expected=2100 if name=='cuda_full' else 300
        assert stats_file['complete'] and check['complete'] and check['completed']==len(raw)==len(rescored)==expected
        assert check['protocol_id']==stats_file['protocol_id']
        by_key={(r['label'],r['idx']):r for r in rescored};assert len(by_key)==expected
        for r in raw:
            saved=by_key[r['label'],r['idx']]
            for scope,ref,pred in [('metrics',r['reference'],r['prediction']),('first_metrics',first_sentence(r['reference']),r['first_prediction'])]:
                recomputed=metric(dll,ref,pred)
                for k,value in recomputed.items():
                    delta=abs(value-saved[scope][k]);maxdiff[scope+'/'+k]=max(maxdiff[scope+'/'+k],delta)
                    assert delta<1e-12,(name,r['label'],r['idx'],scope,k,value,saved[scope][k])
            assert abs(r['phi']-saved['metrics']['phi'])<1e-12
            checked+=1
        print('Independent text audit:',name,len(raw),flush=True)
        rng=np.random.default_rng(stats_file['bootstrap_seed']);groups={};arrays={};facts=[]
        for mi,model in enumerate(stats_file['models']):
            lab=model['label'];rows=sorted([r for r in rescored if r['label']==lab],key=lambda r:r['idx'])
            rawrows=sorted([r for r in raw if r['label']==lab],key=lambda r:r['idx'])
            assert [r['idx'] for r in rows]==list(range(1,301))
            phi=np.asarray([r['metrics']['phi'] for r in rows]);arrays[lab]=phi
            scenarios=sorted({r['scenario'] for r in rows});assert len(scenarios)==60
            groups[lab]=np.array([np.mean([r['metrics']['phi'] for r in rows if r['scenario']==s]) for s in scenarios])
            assert np.allclose(boot(groups[lab],rng),model['phi_ci_scenario_bootstrap'],rtol=0,atol=1e-12)
            for scope in ['metrics','first_metrics']:
                for k in rows[0][scope]:
                    a=[r[scope][k] for r in rows]
                    assert abs(np.mean(a)-model[scope][k]['mean'])<1e-12
                    assert np.allclose(boot(a,rng),model[scope][k]['ci_prompt_bootstrap'],rtol=0,atol=1e-12)
            eos=np.array([r['response_complete'] for r in rows],dtype=bool);ephi=phi[eos]
            k=int(sum(phi>.5));ke=int(sum(ephi>.5));equal=int(sum(phi==.5))
            assert int(sum(phi>.5))==round(model['LR_at_0_5']*300)
            eos_seed=[20261101,ci,mi]
            eos_ci=boot(ephi,np.random.default_rng(eos_seed)) if len(ephi) else [None,None]
            widths={'scenario':model['phi_ci_scenario_bootstrap'][1]-model['phi_ci_scenario_bootstrap'][0],
                    'prompt':model['metrics']['phi']['ci_prompt_bootstrap'][1]-model['metrics']['phi']['ci_prompt_bootstrap'][0]}
            f={'label':lab,'model':model['model'],'n':300,'n_scenarios':60,'diagnostic_series':model['diagnostic_series'],
               'metrics':model['metrics'],'first_metrics':model['first_metrics'],
               'phi_ci_scenario_bootstrap':model['phi_ci_scenario_bootstrap'],
               'scenario_ci_width_change_percent':100*(widths['scenario']/widths['prompt']-1),
               'LR_success_count':k,'LR_percent':100*k/300,'LR_wilson_percent':[100*x for x in wilson(k,300)],'phi_exactly_0_5':equal,
               'EOS_count':int(eos.sum()),'EOS_percent':100*float(eos.mean()),
               'token_limit_count':sum(len(r['token_lengths'])>=512 for r in rawrows),
               'token_limit_percent':100*sum(len(r['token_lengths'])>=512 for r in rawrows)/300,
               'EOS_only_phi_mean':float(ephi.mean()) if len(ephi) else None,
               'EOS_only_phi_ci':eos_ci,'EOS_only_bootstrap_seed_sequence':eos_seed,
               'EOS_only_LR_count':ke,'EOS_only_LR_percent':100*ke/len(ephi) if len(ephi) else None,
               'EOS_only_LR_wilson_percent':[100*x for x in wilson(ke,len(ephi))] if len(ephi) else [None,None],
               'reference_with_ascii_space':sum(' ' in r['reference'] for r in rawrows),
               'reference_with_replacement_character':sum('\ufffd' in r['reference'] for r in rawrows),
               'tokens_mean':float(np.mean([len(r['token_lengths']) for r in rawrows])),
               'reference_characters_mean':float(np.mean([len(r['reference']) for r in rawrows])),
               'token_bytes_mean_of_sample_means':float(np.mean([np.mean(r['token_lengths']) for r in rawrows])),
               'available_segments_sum':sum(r['available_segments'] for r in rawrows),
               'reconstructed_segments_sum':sum(r['reconstructed_segments'] for r in rawrows),
               'available_segments_mean':float(np.mean([r['available_segments'] for r in rawrows])),
               'encoder_calls':sum(len(r['actual_encoder_inputs']) for r in rawrows),
               'truncated_encoder_calls':sum(x['truncated'] for r in rawrows for x in r['actual_encoder_inputs']),
               'responses_with_truncated_encoder_input':sum(any(x['truncated'] for x in r['actual_encoder_inputs']) for r in rawrows),
               'old_phi_mean':model['old_phi_mean'],'delta_phi_mean':model['delta_phi_mean'],
               'topics':model['topics']}
            assert f['EOS_count']==model['eos_count']
            facts.append(f)
        # Reproduce the verifier's paired draw order; significance itself is
        # independently checked through SciPy's signed-rank implementation.
        for prompt_pair,scenario_pair in zip(stats_file['pairwise_prompt_tests'],stats_file['pairwise_scenario_tests']):
            for pair,values in [(prompt_pair,arrays),(scenario_pair,groups)]:
                a,b=values[pair['a']],values[pair['b']]
                assert np.allclose(boot(a-b,rng),pair['diff_ci_bootstrap'],rtol=0,atol=1e-12)
                calc=signed_rank(a,b)
                assert abs(calc['p_raw']-pair['p_raw'])<1e-12
                assert abs(calc['rank_biserial']-pair['rank_biserial'])<1e-12
                assert abs(float((a-b).mean())-pair['mean_diff_phi'])<1e-12
        for key in ['pairwise_prompt_tests','pairwise_scenario_tests']:
            pairs=stats_file[key];running=0.
            for rank,pair in enumerate(sorted(pairs,key=lambda p:p['p_raw'])):
                running=max(running,(len(pairs)-rank)*pair['p_raw'])
                assert abs(min(1.,running)-pair['p_holm'])<1e-12
                assert pair['significant_0_05']==(pair['p_holm']<.05)
        ledger['cohorts'][name]={'protocol_id':stats_file['protocol_id'],'n':expected,'models':facts,
             'pairwise_prompt_tests':stats_file['pairwise_prompt_tests'],'pairwise_scenario_tests':stats_file['pairwise_scenario_tests'],
             'significant_prompt_pairs':sum(p['significant_0_05'] for p in stats_file['pairwise_prompt_tests']),
             'significant_scenario_pairs':sum(p['significant_0_05'] for p in stats_file['pairwise_scenario_tests']),
             'coverage':{k:check[k] for k in ['full_segment_coverage_count','early_stop_count','empty_candidate_stop_count','attempted_segments','available_segments','truncated_encoder_calls']}}
        if name=='cuda_full':ledger['main_covariate_regression']=regression(raw)
        save('main_fresh_factledger.json',ledger)
        print('Means, CIs, thresholds, EOS and paired tests verified:',name,flush=True)
    ledger['independent_raw_pairs_recomputed']=checked
    ledger['max_metric_absolute_difference']=dict(maxdiff)
    ledger['all_checks_passed']=True
    save('main_fresh_factledger.json',ledger)
    print(json.dumps({'all_checks_passed':True,'raw_pairs_recomputed':checked,'max_metric_absolute_difference':dict(maxdiff),'main_covariate_regression':ledger['main_covariate_regression']},indent=2))


if __name__=='__main__':main()
