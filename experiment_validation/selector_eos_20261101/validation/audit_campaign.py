"""Audit all saved EOS candidates and paired changes, without model inference.

Final usage: python3 .../validation/audit_campaign.py
Preflight: same command --smoke-dir .../preflight
Partial: same command --allow-partial --output .../partial_validation
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
OLD = ROOT/'experiment_validation/revalidation_20261031'
sys.path.insert(0,str(ROOT))
from experiment_validation.scripts.generate_prompts import TOPICS
from candidate_checks import audit_record, packed, phi, require

NAMES = ['cuda_full','fresh_tinyllama_cuda','fresh_phi_cuda','secondary']
EXPECTED = dict(cuda_full=2100,fresh_tinyllama_cuda=300,fresh_phi_cuda=300,secondary=3900)
BOOT = 10000


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                require(line.endswith('\n'), 'Incomplete JSONL record: '+str(path))
                yield json.loads(line)


def save(path, value):
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def seed(*parts):
    return int.from_bytes(hashlib.blake2s('\x1f'.join(map(str,parts)).encode(),digest_size=4).digest(),'big')&0x7fffffff


def scenario(source):
    found=[x for x in TOPICS[source['topic']]['issues'] if x.lower() in source['prompt'].lower()]
    require(len(found)==1, 'Ambiguous scenario')
    return source['topic']+'::'+found[0]


def bootstrap(values, rng):
    a=np.asarray(values,dtype=float)
    return np.quantile(a[rng.integers(len(a),size=(BOOT,len(a)))].mean(axis=1),[.025,.975]).tolist()


def crossed_bootstrap(matrix, rng):
    a=np.asarray(matrix,dtype=float); estimates=[]
    for _ in range(BOOT):
        pi=rng.integers(a.shape[0],size=a.shape[0]); si=rng.integers(a.shape[1],size=a.shape[1])
        estimates.append(float(a[pi][:,si].mean()))
    return np.quantile(estimates,[.025,.975]).tolist()


def signed_rank(values):
    d=np.asarray(values,dtype=float); d=d[d!=0]
    if not len(d):
        return dict(p_raw=1.,rank_biserial=0.,nonzero_pairs=0)
    ranks=stats.rankdata(abs(d),method='average')
    return dict(p_raw=float(stats.wilcoxon(d,correction=True,zero_method='wilcox',mode='approx').pvalue),
                rank_biserial=float((ranks[d>0].sum()-ranks[d<0].sum())/ranks.sum()),nonzero_pairs=len(d))


def holm(items, field='p_raw'):
    running=0.
    for rank,item in enumerate(sorted(items,key=lambda x:x[field])):
        running=max(running,(len(items)-rank)*item[field]);item['p_holm']=min(1.,running)
        item['significant_holm_0_05']=item['p_holm']<.05


def paired_analyses(records):
    grouped=defaultdict(list)
    for r in records:
        key=(r['cohort'],r['label']) if r['cohort']!='secondary' else (r['suite'],r['label'],r['condition'],r['attacker'])
        grouped[key].append(r)
    results=[]
    for key,rr in grouped.items():
        rr=sorted(rr,key=lambda r:(r['idx'],r.get('seed_index',0)))
        d=np.array([r['new_phi']-r['old_phi'] for r in rr])
        one=dict(key=list(key),n=len(rr),old_mean_phi=float(np.mean([r['old_phi'] for r in rr])),
                 new_mean_phi=float(np.mean([r['new_phi'] for r in rr])),delta_phi_mean=float(d.mean()),
                 changed_predictions=sum(r['changed_prediction'] for r in rr))
        if rr[0]['cohort']!='secondary':
            scenarios=sorted({r['scenario'] for r in rr})
            scenario_d=np.array([np.mean([r['new_phi']-r['old_phi'] for r in rr if r['scenario']==s]) for s in scenarios])
            prompt_seed=seed('selector-impact-prompt-v1',*key)
            scenario_seed=seed('selector-impact-scenario-v1',*key)
            one.update(n_scenarios=len(scenarios),ci_paired_prompt=bootstrap(d,np.random.default_rng(prompt_seed)),
                       ci_paired_scenario=bootstrap(scenario_d,np.random.default_rng(scenario_seed)),
                       bootstrap_seed_prompt=prompt_seed,bootstrap_seed_scenario=scenario_seed,
                       prompt_test=signed_rank(d),scenario_test=signed_rank(scenario_d),
                       family='archived-seven' if key[0]=='cuda_full' else 'verified-two')
        else:
            ids=sorted({r['idx'] for r in rr}); seeds=sorted({r['seed_index'] for r in rr})
            by={(r['idx'],r['seed_index']):r['new_phi']-r['old_phi'] for r in rr}
            require(len(by)==len(ids)*len(seeds)==len(rr), 'Missing crossing in paired comparison')
            matrix=np.array([[by[i,s] for s in seeds] for i in ids])
            ci_seed=seed('selector-impact-secondary-v1',*key)
            one.update(n_prompts=len(ids),n_decoder_seeds=len(seeds),bootstrap_seed=ci_seed,
                       ci_paired_crossed=crossed_bootstrap(matrix,np.random.default_rng(ci_seed)) if len(seeds)>1 else None,
                       ci_paired_prompt=bootstrap(matrix[:,0],np.random.default_rng(ci_seed)) if len(seeds)==1 else None,
                       prompt_test=signed_rank(matrix.mean(axis=1)),family=key[0])
        results.append(one)
    for family in sorted({r['family'] for r in results}):
        ff=[r for r in results if r['family']==family]
        holm([r['prompt_test'] for r in ff])
        ss=[r['scenario_test'] for r in ff if 'scenario_test' in r]
        if ss:holm(ss)
    return dict(scope='Paired corrected versus original-selector outputs under identical retained references, traces and seeds. End-to-end context can diverge after first selection. Corpus-conditional lexical effects, not disclosure rates.',
                bootstrap_draws=BOOT,seed_recipe='BLAKE2s stable_seed; per-comparison integer seeds retained below.',
                multiplicity='Separate Holm families: seven archived model effects; two verified-reference model effects; eight trace-control cells; twelve defense cells; twenty-two MAP cells. Prompt and scenario tests corrected separately. MAP tests use 30 prompt means, not 150 independent draws.',comparisons=results)


def rendering_analyses(records):
    main={(r['label'],r['idx']):r for r in records if r['cohort']=='cuda_full'}
    rng=np.random.default_rng(20261101); models=[]
    for cohort,label in [('fresh_tinyllama_cuda','tinyllama_1_1b'),('fresh_phi_cuda','phi_3_5_mini')]:
        rr=sorted([r for r in records if r['cohort']==cohort],key=lambda r:r['idx'])
        require(len(rr)==300,'Incomplete paired rendering cohort')
        for r in rr:
            old=main[label,r['idx']]
            require(r['decoder_seed']==old['decoder_seed'] and r['scenario']==old['scenario'],'Rendering pair mismatch')
        d=np.array([r['new_phi']-main[label,r['idx']]['new_phi'] for r in rr])
        scenarios=sorted({r['scenario'] for r in rr})
        sd=np.array([np.mean([r['new_phi']-main[label,r['idx']]['new_phi'] for r in rr if r['scenario']==s]) for s in scenarios])
        models.append(dict(label=label,n=300,n_scenarios=len(scenarios),delta_phi_mean=float(d.mean()),
                           delta_phi_ci_paired_prompt=bootstrap(d,rng),delta_phi_ci_paired_scenario=bootstrap(sd,rng),
                           individual_delta_range=[float(d.min()),float(d.max())]))
    return dict(scope='Matched contextual versus legacy rendering with corrected EOS-bounded selector in both arms. Both length input and scoring reference change. Fixed-ID legacy equality is established by the retained prior reference audit, not newly inferred historical IDs.',
                bootstrap_seed=20261101,bootstrap_draws=BOOT,order='Tiny prompt/scenario then Phi prompt/scenario',models=models,
                fixed_id_evidence_path=str((OLD/'final_review_20261101/fresh_legacy_replay.json').relative_to(ROOT)),
                fixed_id_evidence_sha256=file_sha(OLD/'final_review_20261101/fresh_legacy_replay.json'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--allow-partial',action='store_true')
    ap.add_argument('--output',type=Path,default=HERE)
    ap.add_argument('--smoke-dir',type=Path)
    ap.add_argument('--cohort',action='append',choices=NAMES)
    args=ap.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=True)
    targets=args.cohort or NAMES
    tokenizers={}; original_runs={}; count=Counter(); per_run={}; scalar_records=[]; hashes={}
    hash_cache={}
    def bound(path,expected=None):
        path=Path(path); actual=hash_cache.setdefault(str(path),file_sha(path))
        if expected is not None:require(actual==expected, 'Source hash changed: '+str(path))
        hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]=actual
        return actual
    def original(name):
        if name not in original_runs:
            path=OLD/('secondary/cuda_full' if name=='secondary' else 'inference/'+name)/'samples.jsonl'
            bound(path)
            original_runs[name]={r['task_id'] if name=='secondary' else (r['label'],r['idx']):r for r in rows(path)}
        return original_runs[name]
    def tokens(protocol):
        for model,revision in protocol['checkpoints'].items():
            if model not in tokenizers:
                tokenizers[model]=AutoTokenizer.from_pretrained(model,revision=revision,local_files_only=True,use_fast=True)
        return tokenizers
    try:
        for name in targets:
            is_secondary=name=='secondary'
            if args.smoke_dir:
                if name not in ['cuda_full','secondary']:continue
                run=args.smoke_dir/('secondary_smoke' if is_secondary else 'main_smoke')
                inventory=run/'input_inventory.jsonl' if is_secondary else args.smoke_dir/'canonical_subset.jsonl'
            else:
                run=BASE/('secondary/cuda_full' if is_secondary else 'inference/'+name)
                inventory=run/'input_inventory.jsonl' if is_secondary else (
                    OLD/'collection/canonical_traces.jsonl' if name=='cuda_full' else
                    OLD/'fresh_references'/('tinyllama' if name=='fresh_tinyllama_cuda' else 'phi')/'canonical_traces.jsonl')
            if not (run/'samples.jsonl').exists():
                require(args.allow_partial or args.smoke_dir,'Missing planned output: '+str(run))
                per_run[name]=dict(complete=False,records=0,missing=True);continue
            protocol=json.loads((run/'protocol.json').read_text());bound(run/'protocol.json')
            require(protocol['checkpoints']==json.loads((OLD/'inference/cuda_full/protocol.json').read_text())['checkpoints'], 'Checkpoint revisions changed')
            require(protocol['precision']=='float32' and protocol['torch']=='2.5.1+cu124' and protocol['transformers']=='4.46.3','Execution comparison changed')
            require(protocol['device']=='cuda' and protocol['threads']==4,'Execution device/thread comparison changed')
            bound(inventory,protocol['input_inventory_sha256' if is_secondary else 'input_sha256'])
            binding=protocol['sources'] if is_secondary else protocol.get('original_source_sha256',{})
            for source,h in binding.items():bound(ROOT/source,h)
            if not is_secondary:bound(ROOT/'weiss_reconstruction.py',protocol['reconstructor_sha256'])
            pid=hashlib.sha256((packed(protocol) if is_secondary else json.dumps(protocol,sort_keys=True)).encode()).hexdigest()
            inv={r['task_id'] if is_secondary else (r['label'],r['idx']):r for r in rows(inventory)}
            expected=len(inv);seen=set();counts=Counter()
            if not args.smoke_dir:require(expected==EXPECTED[name],'Unexpected full inventory size')
            for row in rows(run/'samples.jsonl'):
                key=row['task_id'] if is_secondary else (row['label'],row['idx'])
                require(key in inv and key not in seen, 'Unexpected/duplicate record key')
                seen.add(key);require(row['protocol_id']==pid,'Record protocol mismatch')
                origin='secondary' if is_secondary else row['original_selector_run']
                prior=original(origin)[key];source=inv[key]
                evidence=audit_record(row,source,prior,tokens(protocol),secondary=is_secondary)
                counts.update(evidence);count.update(evidence)
                entry=dict(cohort=name,label=row['label'],idx=row['idx'],scenario=scenario(source),
                           decoder_seed=row['decoder_seed' if is_secondary else 'reconstruction_seed'],
                           old_phi=prior['metrics']['phi'] if is_secondary else prior['phi'],
                           new_phi=row['metrics']['phi'] if is_secondary else row['phi'],
                           changed_prediction=row['prediction']!=prior['prediction'])
                if is_secondary:entry.update({k:row[k] for k in ['suite','condition','attacker','seed_index']})
                scalar_records.append(entry)
            status=json.loads((run/'status.json').read_text());bound(run/'status.json')
            require(status['protocol_id']==pid,'Status protocol mismatch')
            complete=len(seen)==expected and status['complete'] is True and status['status']=='completed'
            if status['status']=='completed':require(complete,'False completed status')
            require(complete or args.allow_partial or args.smoke_dir,'Incomplete planned output')
            bound(run/'samples.jsonl')
            per_run[name]=dict(complete=complete,records=len(seen),expected=expected,counts=dict(counts))
            print(f'Audited {name}: {len(seen)}/{expected}; candidates={counts["candidates"]}',flush=True)
        complete=not args.smoke_dir and set(targets)==set(NAMES) and all(per_run[n]['complete'] for n in NAMES)
        if complete:
            require(count['records']==6600 and count['map_predictions_matched']==3300,'Full campaign completeness/negative control')
            paired=paired_analyses(scalar_records);save(out/'corrected_vs_original.json',paired)
            rendering=rendering_analyses(scalar_records);save(out/'paired_rendering_audit.json',rendering)
        report=dict(status='PASS' if complete else ('PREFLIGHT_PASS' if args.smoke_dir else 'PARTIAL_DIAGNOSTIC'),
                    complete=complete,verified_utc=datetime.now(timezone.utc).isoformat(),records_checked=count['records'],
                    map_predictions_matched=count['map_predictions_matched'],counts=dict(count),runs=per_run,
                    all_observed_record_checks_passed=True,source_sha256=hashes,
                    verifier_sha256=file_sha(__file__),candidate_checks_sha256=file_sha(HERE/'candidate_checks.py'),
                    floating_sum_check='Independent double-precision fsum compared with saved CUDA float32 sums at max(1e-5,2e-6*max(1,sum(abs(scores)))) tolerance; saved scalars then reproduce exact stable sorting and duplicate handling.',
                    scope='Candidate-token/EOS/sum/ranking audit, full context chain and text metrics, exact first-call original-selection counterfactual, paired corrected-vs-original outputs. No semantic disclosure certification.')
        save(out/'audit_campaign.json',report)
        print(json.dumps({k:report[k] for k in ['status','complete','records_checked','map_predictions_matched','counts']},indent=2))
        return 0
    except Exception as exc:
        save(out/'audit_campaign.json',dict(status='FAIL',complete=False,records_checked=count['records'],
                 error_type=type(exc).__name__,error=str(exc),runs=per_run))
        raise


if __name__=='__main__':
    raise SystemExit(main())
