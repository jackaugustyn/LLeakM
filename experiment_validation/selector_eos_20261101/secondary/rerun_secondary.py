"""Reproducible new inference for 600 pilot and 3300 MAP records.

The original canonical input inventory, inverse lengths, calibration traces and
all seeds are frozen from the completed 2026-10-31 campaign. Only multi-candidate
selection changes to a sum through first EOS. All 3300 one-candidate MAP records
are rerun as a negative control. Every generated candidate is retained.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LEGACY = ROOT/'experiment_validation/revalidation_20261031'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent/'inference'))
from candidate_evidence import validate_record_candidates, slim_secondary
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import numpy as np
from experiment_validation.scripts import robustness_eval as rb
from experiment_validation.scripts.run_validation import validated_log_frame_lengths
import weiss_reconstruction as wr

PILOT_RUNS = {'qwen_1_5b':'run_20260325_104518_qwen_1_5b',
              'llama_3_2_3b':'run_20260414_073030_llama_3_2_3b'}
LABELS = list(PILOT_RUNS)
CONTROLS = ['real','shuffled','constant','cross_prompt']
CONDITIONS = ['baseline','bucket_8','pad_32','batch_2','batch_4','rand_pad_8']
SEEDS = [20260706,20260707,20260708,20260709,20260710]
EXPECTED = {'trace_control':240,'defense_pilot':360,'map':3300}
REVISIONS = {wr.FIRST_SENTENCES_MODEL:'057cf090b719f5cf89da504babd2122d404e0953',
             wr.MIDDLE_SENTENCES_MODEL:'07e2b3fc96bc129eb4a2871e58731c07e889cecc'}
WORD = re.compile(r"[A-Za-z0-9']+")
BOOT = 10000


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def exact_file(path, text):
    """Create an immutable inventory, or verify the existing bytes on resume."""
    path=Path(path)
    if path.exists():
        if path.read_text()!=text:
            raise ValueError(f'Existing input/provenance differs: {path}; use a new directory')
    else:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)


def validated_source(sample, run_dir, victim_budget, sources):
    path=ROOT/'logs'/f"{sample['run_id']}.json"
    raw=json.loads(path.read_text());sources[str(path.relative_to(ROOT))]=file_sha(path)
    frames=validated_log_frame_lengths(raw,run_id=sample['run_id'],prompt=sample['prompt'])
    clean=wr.frame_lengths_to_token_lengths(frames)
    if (sample['response_text'] != raw['response_text'] or sample['victim_model_id'] != raw['model_id']
        or sample['token_count'] != raw['generated_tokens'] or raw['max_new_tokens'] != victim_budget):
        raise ValueError(f'Source/log inconsistency: {path}')
    if not clean or sum(clean)<=0:
        raise ValueError('Empty source trace')
    return dict(idx=sample['idx'],topic=sample['topic'],prompt=sample['prompt'],run_id=sample['run_id'],
        run_dir=run_dir,model=raw['model_id'],reference=raw['response_text'],
        reference_sha256=hashlib.sha256(raw['response_text'].encode()).hexdigest(),
        generation_log_sha256=sources[str(path.relative_to(ROOT))],clean_lengths=clean,
        clean_lengths_sha256=digest(clean),victim_budget=victim_budget,
        source_response_complete=raw.get('response_complete'),
        reference_scope='validated archived emitted text; ordinary token-sequence detokenization unverified')


def make_item(source, label, suite, condition, attacker, seed_index, decoder_seed, trace_seed,
              observed, attack_input, donor=None):
    if any(type(n) is not int or n<0 for n in observed+attack_input):
        raise ValueError('Invalid transformed trace')
    n=1 if suite=='map' else 3
    item=dict(source,label=label,suite=suite,condition=condition,attacker=attacker,
        seed_index=seed_index,decoder_seed=decoder_seed,trace_seed=trace_seed,
        observed_lengths=observed,attacker_input_lengths=attack_input,
        observed_lengths_sha256=digest(observed),attacker_input_lengths_sha256=digest(attack_input),
        donor=donor,num_first_candidates=n,samples_per_segment=n,max_sentences=n,
        observed_payload_overhead_pct=100*(sum(observed)-sum(source['clean_lengths']))/sum(source['clean_lengths']),
        n_observed_units=len(observed),n_attacker_input_units=len(attack_input))
    item['task_id']=':'.join(map(str,(suite,label,source['idx'],condition,attacker,seed_index)))
    item['input_sha256']=digest(item)
    return item


def build_inventory():
    """Load and verify the exact old inputs; never refit the MAP inverse."""
    old = LEGACY/'secondary/cuda_full'
    old_protocol = json.loads((old/'protocol.json').read_text())
    for filename, key in [('input_inventory.jsonl','input_inventory_sha256'),('calibration.json','calibration_sha256')]:
        if file_sha(old/filename) != old_protocol[key]:
            raise ValueError('Frozen secondary inventory mismatch: '+filename)
    sources = json.loads((old/'sources.json').read_text())
    if sources != old_protocol['sources']:
        raise ValueError('Frozen source inventory differs from original protocol')
    # The selector is the only deliberately changed original dependency. Its
    # original bytes are preserved in this campaign's before/ snapshot.
    original_selector = HERE.parent/'before/weiss_reconstruction.py'
    for name, expected in sources.items():
        path = original_selector if name == 'weiss_reconstruction.py' else ROOT/name
        if file_sha(path) != expected: raise ValueError('Frozen source hash mismatch: '+name)
    sources = dict(sources)
    sources.pop('weiss_reconstruction.py')
    sources[str(original_selector.relative_to(ROOT))] = file_sha(original_selector)
    for filename in ['input_inventory.jsonl','calibration.json','sources.json','plan.json','protocol.json','samples.jsonl']:
        sources[str((old/filename).relative_to(ROOT))] = file_sha(old/filename)
    inventory = rows(old/'input_inventory.jsonl')
    if Counter(item['suite'] for item in inventory) != EXPECTED or len({item['task_id'] for item in inventory}) != 3900:
        raise ValueError('Incomplete frozen secondary design')
    for item in inventory:
        if item['input_sha256'] != digest({k:v for k,v in item.items() if k!='input_sha256'}):
            raise ValueError('Frozen input record digest mismatch')
    calibration = json.loads((old/'calibration.json').read_text())
    audit = json.loads((old/'plan.json').read_text())
    audit.update(selector_version='eos-through-first-inclusive/1',
                 frozen_input_inventory=True,map_inverse_refitted=False,
                 original_selector_protocol_sha256=file_sha(old/'protocol.json'),
                 reference_scope='Frozen references and token-length vectors; no victim generation',
                 candidate_evidence_schema='eos-bounded-candidate-audit/1')
    return inventory,calibration,sources,audit,old_protocol['map_archived_config']


# Exact bit-vector Levenshtein and word LCS, independently checked in the
# archived-statistics revalidation; copied here to avoid import-time writes.
def levenshtein(a,b):
    if len(a)>len(b):a,b=b,a
    if not a:return len(b)
    masks=defaultdict(int)
    for i,c in enumerate(a):masks[c]|=1<<i
    vp,vn,score,last=~0,0,len(a),1<<(len(a)-1)
    for c in b:
        x=masks[c]|vn;d0=(((x&vp)+vp)^vp)|x;hp=vn|~(d0|vp);hn=d0&vp
        score+=bool(hp&last)-bool(hn&last);hp=(hp<<1)|1;vn=d0&hp;vp=(hn<<1)|~(d0|hp)
    return score


def lcs(a,b):
    masks=defaultdict(int)
    for i,x in enumerate(b):masks[x]|=1<<i
    state=0
    for x in a:
        v=masks[x]|state;state=v&~(v-((state<<1)|1))
    return state.bit_count()


def metrics(reference,prediction):
    a,b=WORD.findall(reference.lower()),WORD.findall(prediction.lower());ca,cb=Counter(a),Counter(b)
    denom=math.sqrt(sum(v*v for v in ca.values()))*math.sqrt(sum(v*v for v in cb.values()))
    answer={'phi':sum(ca[k]*cb[k] for k in ca)/denom if denom else 0.,
            'ed_norm':levenshtein(reference,prediction)/max(1,len(reference))}
    for name,n in [('rouge1',sum((ca&cb).values())),('rougeL',lcs(a,b))]:
        p,r=n/len(b) if b else 0.,n/len(a) if a else 0.
        answer[name+'_precision']=p;answer[name+'_recall']=r
        answer[name+'_f1']=2*p*r/(p+r) if p+r else 0.
    return answer


def record_from_result(item,result,actual_inputs,protocol_id,elapsed):
    requested=min(item['max_sentences'],result.available_segment_count)
    record=dict(item,protocol_id=protocol_id,prediction=result.full_text,first_prediction=result.first_sentence,
        reconstructed_segments=result.sentence_count,available_segments=result.available_segment_count,
        requested_segments=requested,attempted_segments=len(actual_inputs),
        stopped_before_requested_segments=result.sentence_count<requested,
        empty_first_prediction=not bool(result.first_sentence.strip()),
        actual_encoder_inputs=list(actual_inputs),metrics=metrics(item['reference'],result.full_text),elapsed_seconds=elapsed)
    record['record_sha256']=digest(record)
    return record


_TOKENIZERS = {}


def pinned_tokenizer(model_id):
    if model_id not in _TOKENIZERS:
        from transformers import AutoTokenizer
        _TOKENIZERS[model_id]=AutoTokenizer.from_pretrained(model_id,revision=REVISIONS[model_id],
            local_files_only=True,use_fast=True)
    return _TOKENIZERS[model_id]


def validate_record(record,item,protocol_id):
    if record.get('protocol_id')!=protocol_id:raise ValueError('Mixed protocol record')
    if any(record.get(k)!=v for k,v in item.items()):raise ValueError('Resumed row/input mismatch')
    without_hash={k:v for k,v in record.items() if k!='record_sha256'}
    if record.get('record_sha256')!=digest(without_hash):raise ValueError('Record digest mismatch')
    if record.get('metrics')!=metrics(item['reference'],record['prediction']):raise ValueError('Saved metric/text mismatch')
    validate_record_candidates(record)
    segments=wr.heuristic_sentences_from_lengths(item['attacker_input_lengths'])
    available=len(segments);requested=min(item['max_sentences'],available)
    actual=record['actual_encoder_inputs'];attempted=len(actual)
    if record['available_segments']!=available or record['requested_segments']!=requested:
        raise ValueError('Segment scope mismatch')
    if record['attempted_segments']!=attempted or attempted>requested or (requested and not attempted):
        raise ValueError('Invalid attempted segment count')
    selected=[]
    for i,x in enumerate(actual):
        model_id=wr.FIRST_SENTENCES_MODEL if i==0 else wr.MIDDLE_SENTENCES_MODEL
        expected_text=wr.make_input_from_lengths(segments[i]) if i==0 else wr.make_input_with_context(segments[i],selected[-1])
        if x['model_id']!=model_id or x['text']!=expected_text:
            raise ValueError('Encoder serialization/context mismatch')
        tokenizer=pinned_tokenizer(model_id)
        full=tokenizer([expected_text],padding=True,truncation=False)['input_ids'][0]
        kept=tokenizer([expected_text],padding=True,truncation=True,max_length=80)['input_ids'][0]
        if x['full_token_ids']!=full or x['retained_token_ids']!=kept or x['full_token_count']!=len(full):
            raise ValueError('Saved encoder IDs differ from pinned tokenizer')
        expected_samples=max(item['num_first_candidates'],item['samples_per_segment']) if i==0 else item['samples_per_segment']
        if x['max_length']!=80 or x['samples']!=expected_samples or x['truncated']!=(len(full)>len(kept)):
            raise ValueError('Encoder sampling/truncation metadata mismatch')
        text=x['selected_text']
        if not isinstance(text,str):raise ValueError('Selected decoder output is not text')
        if i>0 and not text and i!=attempted-1:
            raise ValueError('Decoding continued after an empty middle output')
        selected.append(text)
    if attempted<requested and not (attempted>=2 and selected[-1]==''):
        raise ValueError('Missing decoder calls without an empty middle output')
    completed=(1+sum(bool(text) for text in selected[1:])) if selected else 0
    if record['reconstructed_segments']!=completed:
        raise ValueError('Reconstructed count differs from selected outputs')
    if record['prediction']!=wr._concat_segments(selected) or record['first_prediction']!=(selected[0] if selected else ''):
        raise ValueError('Saved prediction differs from selected decoder outputs')
    if bool(record['stopped_before_requested_segments'])!=(completed<requested):
        raise ValueError('Incorrect segment completeness flag')
    if record['empty_first_prediction']!=(not bool(record['first_prediction'].strip())):
        raise ValueError('Incorrect empty-first flag')


def load_completed(path,inventory,protocol_id,original):
    by_id={item['task_id']:item for item in inventory};done={}
    if not Path(path).exists():return done
    for line in Path(path).open():
        if not line.strip(): continue
        record=json.loads(line)
        key=record['task_id']
        if key not in by_id or key in done:raise ValueError('Foreign or duplicate resumed record')
        validate_record(record,by_id[key],protocol_id)
        prior=original[key]
        for old_field,new_field in [('prediction','original_selector_prediction'),('phi','original_selector_phi'),('record_sha256','original_selector_record_sha256')]:
            if record.get(new_field)!=prior[old_field]: raise ValueError('Original comparison record mismatch')
        if record['suite']=='map' and record['prediction']!=prior['prediction']:
            raise ValueError('Resumed MAP negative control differs from original output')
        done[key]=slim_secondary(record)
    return done


def bootstrap(values,seed):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    return np.percentile(values[rng.integers(0,len(values),size=(BOOT,len(values)))].mean(axis=1),[2.5,97.5]).tolist()


def signed_rank(a,b):
    from scipy.stats import rankdata
    d=np.asarray(a)-np.asarray(b);d=d[d!=0];n=len(d)
    if not n:return dict(n=0,p_value=1.,rank_biserial=0.)
    ranks=rankdata(np.abs(d));pos=float(ranks[d>0].sum());neg=float(ranks[d<0].sum())
    _,counts=np.unique(np.abs(d),return_counts=True)
    variance=(n*(n+1)*(2*n+1)-float(np.sum(counts**3-counts))/2)/24
    w=min(pos,neg);mean=n*(n+1)/4
    z=(w-mean+0.5)/math.sqrt(variance) if w<mean and variance>0 else 0.
    return dict(n=n,p_value=math.erfc(abs(z)/math.sqrt(2)),rank_biserial=(pos-neg)/(pos+neg))


def aggregate(records,inventory,protocol_id):
    expected=defaultdict(set);cells=defaultdict(list)
    for item in inventory:expected[(item['suite'],item['label'],item['condition'],item['attacker'])].add(item['task_id'])
    for record in records:cells[(record['suite'],record['label'],record['condition'],record['attacker'])].append(record)
    summaries=[];matrices={}
    for key,ids in expected.items():
        rr=cells[key];complete={r['task_id'] for r in rr}==ids and len(rr)==len(ids)
        row=dict(suite=key[0],label=key[1],condition=key[2],attacker=key[3],expected=len(ids),n=len(rr),complete=complete)
        if rr:
            row['metric_means']={name:float(np.mean([r['metrics'][name] for r in rr])) for name in rr[0]['metrics']}
            row['observed_payload_overhead_pct_mean']=float(np.mean([r['observed_payload_overhead_pct'] for r in rr]))
            row['observed_units_mean']=float(np.mean([r['n_observed_units'] for r in rr]))
            row['lr_at_0_5_pct']=100*sum(r['metrics']['phi']>.5 for r in rr)/len(rr)
            row['stopped_before_requested_count']=sum(r['stopped_before_requested_segments'] for r in rr)
            row['empty_first_prediction_count']=sum(r['empty_first_prediction'] for r in rr)
        if complete:
            if key[0]=='map':
                ii=sorted({r['idx'] for r in rr});vv={(r['idx'],r['seed_index']):r['metrics']['phi'] for r in rr}
                matrix=np.array([[vv[(i,s)] for s in SEEDS] for i in ii]);matrices[key]=(ii,SEEDS,matrix)
                row['phi_ci_crossed']=rb.crossed_bootstrap_ci(matrix,rb.stable_seed('bootstrap',*key),BOOT)
                row['between_seed_sd']=float(np.std(matrix.mean(axis=0),ddof=1))
            else:
                rr=sorted(rr,key=lambda r:r['idx']);arr=np.array([r['metrics']['phi'] for r in rr])
                matrices[key]=([r['idx'] for r in rr],[20260706],arr[:,None])
                row['phi_ci_prompt']=bootstrap(arr,rb.stable_seed('bootstrap',*key))
        summaries.append(row)
    suite_complete={suite:all(r['complete'] for r in summaries if r['suite']==suite) for suite in EXPECTED}
    contrasts=[]
    for key,(idxs,seeds,matrix) in matrices.items():
        if not suite_complete[key[0]]:continue
        base_condition='real' if key[0]=='trace_control' else 'baseline';base=(key[0],key[1],base_condition,'frozen')
        if key==base or base not in matrices:continue
        bi,bs,bm=matrices[base]
        if idxs!=bi or seeds!=bs:raise ValueError('Contrast pairing mismatch')
        diff=matrix-bm
        entry=dict(suite=key[0],label=key[1],condition=key[2],attacker=key[3],
                   mean_phi_difference_vs_baseline=float(diff.mean()),
                   score_reduction_pct=100*(1-float(matrix.mean())/float(bm.mean())) if bm.mean()>0 else None)
        if key[0]=='map':entry['ci_crossed']=rb.crossed_bootstrap_ci(diff,rb.stable_seed('contrast',*key),BOOT)
        else:
            entry['ci_prompt']=bootstrap(diff[:,0],rb.stable_seed('contrast',*key))
            entry.update(signed_rank(matrix[:,0],bm[:,0]))
        contrasts.append(entry)
    for suite in ['trace_control','defense_pilot']:
        family=[r for r in contrasts if r['suite']==suite];running=0.
        for rank,row in enumerate(sorted(family,key=lambda r:r['p_value'])):
            running=max(running,(len(family)-rank)*row['p_value']);row['p_holm']=min(1.,running)
    suite_complete={suite:all(r['complete'] for r in summaries if r['suite']==suite) for suite in EXPECTED}
    return dict(protocol_id=protocol_id,complete=all(suite_complete.values()),n=len(records),expected=3900,
        suite_complete=suite_complete,cells=summaries,paired_contrasts=contrasts,
        interpretation='New inference conditional on validated archived emitted-text references. Partial cells have descriptive means only; no intervals. Segment shortfalls are reported, not repaired.',
        reference_scope='No new victim generation; ordinary historical token-sequence decoding remains unverified.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--limit',type=int,default=0,help='New records this invocation; 0=all pending; does not change protocol')
    ap.add_argument('--plan',action='store_true')
    ap.add_argument('--aggregate-only',action='store_true',help='Validate saved rows and regenerate statistics without loading models or requiring CUDA access')
    ap.add_argument('--require-complete',action='store_true',help='Return exit code2 if the validated design remains incomplete')
    args=ap.parse_args()
    if args.threads<1 or args.limit<0:raise ValueError('Invalid threads/limit')
    out=args.output.resolve()
    for protected in [ROOT/'experiment_validation/results',ROOT/'experiment_validation/analysis',LEGACY]:
        if out==protected or protected in out.parents:raise ValueError('Historical output path is protected')
    inventory,calibration,sources,audit,old_config=build_inventory()
    for path in [Path(__file__),Path(rb.__file__),ROOT/'weiss_reconstruction.py',ROOT/'experiment_validation/scripts/run_validation.py',HERE.parent/'inference/candidate_evidence.py']:
        sources[str(path.resolve().relative_to(ROOT))]=file_sha(path)
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock_handle=(out/'.run.lock').open('a')
    fcntl.flock(lock_handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    inventory_text=''.join(packed(r)+'\n' for r in inventory)
    exact_file(out/'input_inventory.jsonl',inventory_text)
    exact_file(out/'calibration.json',packed(calibration)+'\n')
    exact_file(out/'sources.json',packed(sources)+'\n')
    write_json(out/'plan.json',audit)
    if args.plan:
        print(json.dumps(audit,indent=2));return 0
    import torch,transformers,scipy
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
    if not args.aggregate_only and args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    checkpoint_files={m:{name:file_sha(hf_hub_download(m,name,revision=rev,local_files_only=True))
        for name in ['pytorch_model.bin','tokenizer.json','config.json','tokenizer_config.json','special_tokens_map.json','generation_config.json']} for m,rev in REVISIONS.items()}
    old_protocol=json.loads((LEGACY/'secondary/cuda_full/protocol.json').read_text())
    if checkpoint_files != old_protocol['checkpoint_files_sha256']:
        raise ValueError('Checkpoint files differ from frozen source protocol')
    if torch.__version__ != old_protocol['torch'] or transformers.__version__ != old_protocol['transformers'] or args.threads != old_protocol['threads']:
        raise ValueError('Software/thread configuration differs from original campaign')
    protocol=dict(schema_version=2,selector_version='eos-through-first-inclusive/1',scope=audit,input_inventory_sha256=file_sha(out/'input_inventory.jsonl'),
        calibration_sha256=file_sha(out/'calibration.json'),sources=sources,checkpoints=REVISIONS,
        checkpoint_files_sha256=checkpoint_files,device=args.device,precision='float32',threads=args.threads,
        python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
        numpy=np.__version__,scipy=scipy.__version__,bootstrap_resamples=BOOT,encoder_decoder_max_length=80,
        map_archived_config=old_config,seed_source='explicit per-record decoder and transformation seeds')
    protocol_id=digest(protocol)
    exact_file(out/'protocol.json',packed(protocol)+'\n')
    original={}
    with (LEGACY/'secondary/cuda_full/samples.jsonl').open() as source:
        for line in source:
            if not line.strip(): continue
            record=json.loads(line);key=record['task_id']
            if key in original: raise ValueError('Duplicate original secondary record')
            original[key]={'prediction':record['prediction'],'phi':record['metrics']['phi'],
                           'record_sha256':record['record_sha256']}
    if set(original)!={item['task_id'] for item in inventory}: raise ValueError('Original comparison key mismatch')
    samples=out/'samples.jsonl';done=load_completed(samples,inventory,protocol_id,original)
    status=dict(protocol_id=protocol_id,expected=3900,completed=len(done),complete=len(done)==3900,
        status='starting',pid=os.getpid(),new_records_this_invocation=0,device=args.device)
    write_json(out/'status.json',status);start=time.monotonic();actual=[]
    def load_bundle(model_id):
        if model_id not in wr._MODEL_CACHE:
            rev=REVISIONS[model_id]
            tokenizer=AutoTokenizer.from_pretrained(model_id,revision=rev,local_files_only=True,use_fast=True)
            model=AutoModelForSeq2SeqLM.from_pretrained(model_id,revision=rev,local_files_only=True,
                low_cpu_mem_usage=True,torch_dtype=torch.float32)
            model.to(args.device).eval();wr._MODEL_CACHE[model_id]=(model,tokenizer,args.device)
        return wr._MODEL_CACHE[model_id]
    wr._load_model_bundle=load_bundle;original_sample=wr._sample_and_rank
    def observed_sample(model,tokenizer,device,input_text,*,samples,max_length=80):
        full=tokenizer([input_text],padding=True,truncation=False)['input_ids'][0]
        kept=tokenizer([input_text],padding=True,truncation=True,max_length=max_length)['input_ids'][0]
        entry=dict(text=input_text,full_token_ids=full,full_token_count=len(full),retained_token_ids=kept,
            truncated=len(full)>len(kept),max_length=max_length,samples=samples,
            model_id=next(k for k,v in wr._MODEL_CACHE.items() if v[0] is model))
        actual.append(entry)
        candidate_audit=[]
        ranked=original_sample(model,tokenizer,device,input_text,samples=samples,max_length=max_length,
                               candidate_audit=candidate_audit)
        if len(candidate_audit)!=1: raise ValueError('One audit object required per decoder call')
        entry['candidate_audit']=candidate_audit[0]
        entry['selected_text']=ranked[0][0] if ranked else ''
        return ranked
    wr._sample_and_rank=observed_sample
    try:
        if not args.aggregate_only and len(done)<3900:
            status['status']='loading_pinned_models';write_json(out/'status.json',status)
            for model_id in REVISIONS:load_bundle(model_id)
            for item in inventory:
                if item['task_id'] in done:continue
                if args.limit and status['new_records_this_invocation']>=args.limit:break
                status.update(status='running',active=item['task_id']);write_json(out/'status.json',status)
                actual.clear();started=time.monotonic()
                result=wr.reconstruct(item['attacker_input_lengths'],num_first_candidates=item['num_first_candidates'],
                    samples_per_segment=item['samples_per_segment'],max_sentences=item['max_sentences'],seed=item['decoder_seed'])
                record=record_from_result(item,result,actual,protocol_id,time.monotonic()-started)
                prior=original[item['task_id']]
                record.update(original_selector_prediction=prior['prediction'],original_selector_phi=prior['phi'],
                              original_selector_record_sha256=prior['record_sha256'])
                if item['suite']=='map' and record['prediction']!=prior['prediction']:
                    raise ValueError('Single-candidate MAP negative control differs from the original output')
                record['record_sha256']=digest({k:v for k,v in record.items() if k!='record_sha256'})
                validate_record(record,item,protocol_id)
                with samples.open('a') as handle:
                    handle.write(packed(record)+'\n');handle.flush();os.fsync(handle.fileno())
                done[item['task_id']]=slim_secondary(record);status['new_records_this_invocation']+=1
                status.update(completed=len(done),complete=len(done)==3900,elapsed_seconds=time.monotonic()-start)
                write_json(out/'status.json',status)
                print(f"{len(done)}/3900 {item['task_id']} phi={record['metrics']['phi']:.6f} seconds={record['elapsed_seconds']:.1f}",flush=True)
        status['status']='aggregating';status.pop('active',None);write_json(out/'status.json',status)
        report=aggregate(list(done.values()),inventory,protocol_id)
        write_json(out/'summary.json',report)
        status.update(status='completed' if report['complete'] else 'partial',complete=report['complete'],
                      elapsed_seconds=time.monotonic()-start)
        write_json(out/'status.json',status)
    except BaseException as exc:
        status.update(status='failed',error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-start)
        write_json(out/'status.json',status);raise
    return 0 if report['complete'] or not args.require_complete else 2


if __name__=='__main__':
    raise SystemExit(main())
