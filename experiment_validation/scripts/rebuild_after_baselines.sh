#!/usr/bin/env bash
set -u
cd /home/paugustynowicz/LLeakM
source .venv/bin/activate 2>/dev/null
echo "[watch] waiting for defense_eval and trace_baselines to finish..."
while pgrep -f 'defense_eval.py --per-topic' >/dev/null || pgrep -f 'trace_baselines.py' >/dev/null; do
  sleep 90
done
echo "[watch] jobs finished; aggregating"
python experiment_validation/scripts/defense_eval.py --aggregate-only || true
python experiment_validation/scripts/analyze_extras.py || true
# refresh defense table with pad_32 if present
python3 <<'PY'
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('.'); OUT=ROOT/'experiment_validation/analysis'; RESULTS=ROOT/'experiment_validation/results'; LOGDIR=ROOT/'logs'; SEED=20260706

def wilson(k,n,z=1.959963984540054):
    if n==0: return 0,0,0
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=(z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
    return p,max(0,c-h),min(1,c+h)
def toks(rid):
    p=LOGDIR/f'{rid}.json'
    if not p.exists(): return None
    return [int(s['token_utf8_len']) for s in json.loads(p.read_text()).get('steps',[])]
def apply(dname,t,rng):
    if dname=='baseline': return list(t)
    if dname=='bucket_8': return [int(math.ceil(x/8)*8) for x in t]
    if dname in ('pad_32','pad_fixed'):
        if dname=='pad_32': return [32]*len(t)
        c=max(max(t),16); return [c]*len(t)
    if dname=='batch_2': return [sum(t[i:i+2]) for i in range(0,len(t),2)]
    if dname=='batch_4': return [sum(t[i:i+4]) for i in range(0,len(t),4)]
    if dname=='rand_pad_8': return [int(x+rng.integers(0,9)) for x in t]
    return list(t)

RUNS=[('qwen_1_5b','Qwen2.5-1.5B-Instruct','run_20260325_104518_qwen_1_5b'),
      ('llama_3_2_3b','Llama-3.2-3B-Instruct','run_20260414_073030_llama_3_2_3b')]
meta={}
for lab,_,rd in RUNS:
    m={}
    for line in (RESULTS/rd/'samples.jsonl').open():
        o=json.loads(line); m[int(o['idx'])]=o
    meta[lab]=m
rows=defaultdict(list)
for line in (OUT/'defense_samples.jsonl').open():
    o=json.loads(line); rows[(o['label'], o['defense'])].append(o)
# prefer pad_32 over pad_fixed
order=['baseline','bucket_8','pad_32','pad_fixed','batch_2','batch_4','rand_pad_8']
rng=np.random.default_rng(SEED)
lines=['% Defense table refreshed after pad_32']
prev=None; seen_pad=set()
for lab,pretty,_ in RUNS:
    base=np.array([float(r['phi']) for r in rows[(lab,'baseline')]])
    for dname in order:
        if dname=='pad_fixed' and rows.get((lab,'pad_32')):
            continue
        rs=rows.get((lab,dname),[])
        if not rs: continue
        phi=np.array([float(r['phi']) for r in rs]); n=len(phi); k=int((phi>0.5).sum())
        lr,lo,hi=wilson(k,n)
        bidx=rng.integers(0,n,size=(5000,n)); plo,phi_hi=np.percentile(phi[bidx].mean(axis=1),[2.5,97.5])
        overs=[]
        for r in rs:
            o=meta[lab].get(int(r['idx']))
            if not o: continue
            t=toks(o['run_id'])
            if not t: continue
            seed=(SEED+int(r['idx'])*131+hash(dname)%9973)&0x7fffffff
            defended=apply(dname if dname!='pad_fixed' else 'pad_32', t, np.random.default_rng(seed)) if dname!='pad_fixed' else apply('pad_fixed',t,np.random.default_rng(seed))
            if dname=='pad_32':
                defended=apply('pad_32',t,np.random.default_rng(seed))
            bb,bd=sum(t),sum(defended); overs.append(100*(bd-bb)/bb if bb else 0)
        oh=float(np.mean(overs)) if overs else float('nan')
        red=100*(1-phi.mean()/base.mean()) if base.mean()>0 else 0
        dshow='pad\\_32' if dname=='pad_32' else ('pad\\_32 (look-ahead)' if dname=='pad_fixed' else dname.replace('_','\\_'))
        if pretty!=prev and prev is not None: lines.append('\\midrule')
        cell=pretty if pretty!=prev else ''; prev=pretty
        lines.append(f"{cell} & {dshow} & {n} & {phi.mean():.4f} & $[{plo:.4f}, {phi_hi:.4f}]$ & "
                     f"{100*lr:.1f} $[{100*lo:.1f}, {100*hi:.1f}]$ & {oh:.1f} & {red:.1f} \\\\")
(OUT/'defense_tables_body.tex').write_text('\n'.join(lines)+'\n')
print('defense table updated')
PY
# note: tables are inlined in tifs_main.tex — print reminder
echo "[watch] REBUILD_PARTIAL_COMPLETE — re-inline tables into tifs_main.tex if needed"
echo "[watch] baseline json:"; ls -la experiment_validation/analysis/trace_baselines.json 2>/dev/null || true
