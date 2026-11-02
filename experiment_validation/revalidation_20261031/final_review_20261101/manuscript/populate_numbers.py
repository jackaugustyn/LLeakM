from pathlib import Path
import json,re,hashlib,difflib
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=HERE.parents[3]
s=(HERE/'prose_stage.tex').read_text();before=(HERE/'before/template.tex').read_text()
facts=json.loads((HERE.parent/'main_fresh_factledger.json').read_text())
secondary=json.loads((BASE/'secondary/cuda_full/summary.json').read_text())
secondary_audit=json.loads((HERE.parent/'secondary_audit.json').read_text())
main={r['label']:r for r in facts['cohorts']['cuda_full']['models']}
order=['qwen_1_5b','qwen_3b','phi_1_5','llama_3_2_3b','gemma_2_2b','phi_3_5_mini','tinyllama_1_1b']
name={'qwen_1_5b':'Qwen2.5-1.5B-Instruct','qwen_3b':'Qwen2.5-3B-Instruct','phi_1_5':'Phi-1.5','llama_3_2_3b':'Llama-3.2-3B-Instruct','gemma_2_2b':'Gemma-2-2B-it','phi_3_5_mini':'Phi-3.5-mini-instruct','tinyllama_1_1b':'TinyLlama-1.1B-Chat'}
short={'qwen_1_5b':'Qwen2.5-1.5B','qwen_3b':'Qwen2.5-3B','phi_1_5':'Phi-1.5','llama_3_2_3b':'Llama-3.2-3B','gemma_2_2b':'Gemma-2-2B','phi_3_5_mini':'Phi-3.5-mini','tinyllama_1_1b':'TinyLlama-1.1B'}
fmt=lambda x:'--' if x is None else f'{x:.4f}'
ci=lambda x,d=4:f'[{x[0]:.{d}f}, {x[1]:.{d}f}]'
row=lambda xs:' & '.join(xs)+r' \\'
receipt={}
def body(label,lines):
 global s
 pos=s.index(r'\label{'+label+'}');a=s.index(r'\midrule',pos)+len(r'\midrule');b=s.index(r'\bottomrule',a)
 s=s[:a]+'\n'+'\n'.join(lines)+'\n'+s[b:];receipt[label]=lines
lines=[]
for k in order:
 m=main[k];metrics=m['metrics']
 if k=='phi_3_5_mini':lines.extend([r'\midrule',r'\multicolumn{7}{l}{\emph{Reference-format diagnostics}} \\'])
 lines.append(row([name[k],'base' if k=='phi_1_5' else 'chat' if k=='tinyllama_1_1b' else 'instruct']+[fmt(metrics[x]['mean']) for x in ['phi','ed_norm','rouge1_precision','rougeL_precision']]+[f"{m['LR_percent']:.2f}\\% "]))
body('tab:main_results',lines)
body('tab:statistical_results',[row([name[k],fmt(main[k]['metrics']['phi']['mean']),ci(main[k]['metrics']['phi']['ci_prompt_bootstrap']),f"{main[k]['LR_percent']:.2f}\\%",ci(main[k]['LR_wilson_percent'],2)+r'\%']) for k in order])
body('tab:scenario_sensitivity',[row([short[k]+(r'$^{\dagger}$' if main[k]['diagnostic_series'] else ''),fmt(main[k]['metrics']['phi']['mean']),ci(main[k]['phi_ci_scenario_bootstrap'])]) for k in order])
body('tab:eos_completeness',[row([name[k],str(main[k]['EOS_count']),f"{main[k]['EOS_percent']:.1f}",f"{main[k]['token_limit_percent']:.1f}",fmt(main[k]['metrics']['phi']['mean']),fmt(main[k]['EOS_only_phi_mean']),f"{main[k]['LR_percent']:.2f}",'--' if main[k]['EOS_only_LR_percent'] is None else f"{main[k]['EOS_only_LR_percent']:.2f}"]) for k in order])
cells={(r['suite'],r['label'],r['condition'],r['attacker']):r for r in secondary['cells']}
contrasts={(r['suite'],r['label'],r['condition'],r['attacker']):r for r in secondary['paired_contrasts']}
lines=[]
for label in ['qwen_1_5b','llama_3_2_3b']:
 if lines:lines.append(r'\midrule')
 for condition in ['real','shuffled','constant','cross_prompt']:
  key=('trace_control',label,condition,'frozen');r=cells[key]
  delta='--' if condition=='real' else '$'+f"{contrasts[key]['mean_phi_difference_vs_baseline']:+.4f}"+'$'
  lines.append(row([short[label] if condition=='real' else '',{'real':'true','cross_prompt':'cross-prompt'}.get(condition,condition),fmt(r['metric_means']['phi']),ci(r['phi_ci_prompt']),delta]))
body('tab:trace_controls',lines)
lines=[]
for label in ['qwen_1_5b','llama_3_2_3b']:
 if lines:lines.append(r'\midrule')
 for condition in ['baseline','bucket_8','pad_32','batch_2','batch_4','rand_pad_8']:
  key=('defense_pilot',label,condition,'frozen');r=cells[key]
  reduction='--' if condition=='baseline' else f"{contrasts[key]['score_reduction_pct']:.1f}\\%"
  lines.append(row([short[label] if condition=='baseline' else '',condition.replace('_',r'\_'),fmt(r['metric_means']['phi']),ci(r['phi_ci_prompt']),reduction,f"{r['observed_payload_overhead_pct_mean']:.1f}\\%",f"{r['observed_units_mean']:.0f}"]))
body('tab:defense_results',lines)
lines=[]
for label in ['qwen_1_5b','llama_3_2_3b']:
 if lines:lines.append(r'\midrule')
 for condition in ['baseline','bucket_8','pad_32','batch_2','batch_4','rand_pad_8']:
  for attacker in ['frozen'] if condition=='baseline' else ['frozen','adaptive']:
   key=('map',label,condition,attacker);r=cells[key];red=0.0 if condition=='baseline' else contrasts[key]['score_reduction_pct']
   lines.append(row([short[label] if condition=='baseline' else '',condition.replace('_',r'\_'),attacker,fmt(r['metric_means']['phi']),ci(r['phi_ci_crossed']),fmt(r['between_seed_sd']),f'${red:.1f}$' if red<0 else f'{red:.1f}']))
body('tab:robustness',lines)
# New-reference table, kept separate from all archived reference rows.
fresh=[facts['cohorts'][k]['models'][0] for k in ['fresh_tinyllama_cuda','fresh_phi_cuda']]
fresh_rows=[row([short[m['label']],fmt(m['metrics']['phi']['mean']),ci(m['phi_ci_scenario_bootstrap']),fmt(m['metrics']['ed_norm']['mean']),fmt(m['metrics']['rouge1_precision']['mean']),fmt(m['metrics']['rougeL_precision']['mean']),f"{m['LR_percent']:.2f}\\%",f"{m['EOS_percent']:.1f}\\%"]) for m in fresh]
fresh_table=r'''\begin{table*}[ht]
\centering
\caption{Separate cohorts with newly generated, verified full-sequence references ($N=300$ per model; 512-token victim limit; all segments and three T5 samples per segment). Mean-$\phi$ intervals resample 60 scenarios. The rows are not pooled with the archived cohort because historical victim revision identity is unverified.}
\label{tab:fresh_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c c c c c c c}
\toprule
Model & Mean $\phi$ & Scenario 95\% CI & $ed_{\mathrm{norm}}$ & R1 Precision & RL Precision & $\mathrm{LR}@0.5$ & EOS \\
\midrule
'''+ '\n'.join(fresh_rows)+r'''
\bottomrule
\end{tabular}%
}
\end{table*}

'''
s=s.replace(r'\subsection{Effects of Framing Transformations',fresh_table+r'\subsection{Effects of Framing Transformations',1);receipt['tab:fresh_results']=fresh_rows
# Asymmetric error bars from exact verified mean/CI values.
a=s.index(r'\addplot+[only marks,mark=*,blue,error bars/.cd,y dir=both,y explicit]');b=s.index(r'\end{axis}',a)
symbols=['Qwen1.5','Qwen3','Phi1.5','Llama3','Gemma2','Phi3.5','TinyLlama']
plot=[r'\addplot+[only marks,mark=*,blue,error bars/.cd,y dir=both,y explicit]',r'table[x=model,y=mean,y error plus=plus,y error minus=minus] {','model mean plus minus']
for k,symbol in zip(order,symbols):
 m=main[k]['metrics']['phi'];mean=m['mean'];lo,hi=m['ci_prompt_bootstrap'];plot.append(f'{symbol} {mean:.6f} {hi-mean:.6f} {mean-lo:.6f}')
plot.extend(['};',''])
s=s[:a]+'\n'.join(plot)+s[b:];receipt['fig:model_ci']=plot
# Realized cost and mean per-model reduction, not a ratio of pooled scores.
a=s.index(r'\addplot+[only marks,mark=*,mark size=2.6pt,blue]');b=s.index(r'\end{axis}',a)
fig=secondary_audit['defense_figure_equal_model_average'];coords=' '.join(f"({r['payload_overhead_pct']:.1f},{r['mean_model_score_reduction_pct']:.1f})" for r in fig)
plot=[r'\addplot+[only marks,mark=*,mark size=2.6pt,blue]',f'coordinates {{{coords}}};',
 r'\node[font=\scriptsize,anchor=south] at (axis cs:85.5,87.6) {bucket\_8};',
 r'\node[font=\scriptsize,anchor=south east] at (axis cs:551.6,37.0) {pad\_32};',
 r'\node[font=\scriptsize,anchor=west] at (axis cs:95,78) {batch\_2};',
 r'\draw[gray,thin] (axis cs:0,71.3) -- (axis cs:90,78);',
 r'\node[font=\scriptsize,anchor=west] at (axis cs:95,48) {batch\_4};',
 r'\draw[gray,thin] (axis cs:0,68.6) -- (axis cs:90,48);',
 r'\node[font=\scriptsize,anchor=west] at (axis cs:95,58.5) {rand\_pad\_8};','']
s=s[:a]+'\n'.join(plot)+s[b:];receipt['fig:defense_tradeoff']=plot
# Consolidated factual and stylistic corrections after the independent audits.
s=s.replace('These differ from the 512-token primary campaign in both source traces and collection path.', 'These use different source responses and a lower victim-token budget than the primary cohort, with the same validated log-length representation.')
s=s.replace('below the prespecified minimum of 200','below the configured minimum of 200')
s=s.replace('the primary seven-model table uses one assembled reconstruction','the primary seven-model table uses one assembled reconstruction')
s=s.replace('mean $\\phi$ changes by $-0.020$ to $-0.035$','mean $\\phi$ changes by $-0.035$ to $-0.021$')
s=s.replace('All seven scenario-bootstrap intervals are wider than their prompt-bootstrap counterparts.', 'All seven scenario-bootstrap intervals are wider than their prompt-bootstrap counterparts, by up to 41.6\\% for Qwen2.5-3B.')
s=s.replace('The design comprises 22 complete cells and 3300 reconstructions. All 22 cells contain their full 30-prompt by five-seed design.', 'The design comprises 22 complete cells, each with 30 prompts and five decoder seeds.')
needle='The new results do not isolate a causal detokenization effect.'
# Insert the controlled fixed-ID result as its own paragraph in the audit.
anchor=r'\subsection{Reconstruction Pipeline}'
matched=r'''Holding the newly recorded token IDs fixed, replay of the historical independent-token decoding reproduces every archived fragment, complete response and byte-length vector for all 300 TinyLlama and all 300 Phi-3.5-mini samples. Switching to complete-sequence decoding changes only ASCII word-boundary spaces in these records. This controlled replay identifies a detokenization mechanism for the observed formatting defect. It does not recover the unknown historical IDs or certify old checkpoint versions, and it does not isolate the effect on the full T5 reconstruction score.

'''
s=s.replace(anchor,matched+anchor,1)
s=s.replace('Their corresponding mean scores are 0.4352 and 0.5042', 'Their corresponding mean scores are 0.4352 and 0.5042')
# State actual T5 sampling without calling it nucleus sampling.
s=s.replace('Multiple decoding candidates are ranked by the transition log-probability sum.', 'T5 generation uses stochastic sampling with top-$k=50$, temperature 1 and a two-token no-repeat n-gram constraint. Candidates are ranked by the transition log-probability sum.')
# A concise result of actual independent rerun, not a blanket exact-history claim.
anchor='These findings describe a sensitivity analysis of the evaluated pipeline.'
s=s.replace(anchor,'All 3300 MAP predictions exactly match the retained earlier predictions when aligned by model, prompt, condition, attacker and decoder seed. The current table uses newly recomputed crossed-bootstrap intervals with explicitly bound bootstrap seeds. '+anchor,1)
# Avoid a future-manifest claim: root will produce a post-integration manifest.
s=s.replace('a subsequent manuscript snapshot has a separate role.', 'it is an experiment inventory, not a hash inventory of the integrated manuscript.')
# Published paths refer to accompanying artifact organization; no external availability claim was refreshed.
assert re.findall(r'\\cite\w*\{[^}]+\}',s)==re.findall(r'\\cite\w*\{[^}]+\}',before),'Citation keys/order changed'
for cmd in ['Title','Author','AuthorNames','address','corres','authorcontributions','funding','institutionalreview','informedconsent','conflictsofinterest']:
 a=[line for line in before.splitlines() if line.startswith('\\'+cmd+'{') or line.startswith('\\'+cmd+'[')]
 b=[line for line in s.splitlines() if line.startswith('\\'+cmd+'{') or line.startswith('\\'+cmd+'[')]
 assert a==b,cmd
PAPER=ROOT/'article1/template.tex';PAPER.write_text(s)
(HERE/'numeric_rows.json').write_text(json.dumps(receipt,indent=2)+'\n')
(HERE/'integrated.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='before/template.tex',tofile='article1/template.tex')))
(HERE/'integration_receipt.json').write_text(json.dumps({'status':'source_integrated_pending_compile_and_independent_table_check','template_sha256':hashlib.sha256(PAPER.read_bytes()).hexdigest(),'numeric_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE.parent/'main_fresh_factledger.json',HERE.parent/'secondary_audit.json',BASE/'secondary/cuda_full/summary.json']},'table_labels':list(receipt),'citations_preserved':True,'authors_and_declarations_preserved':True,'pdf_not_built':True},indent=2)+'\n')
print('INTEGRATED',len(s),'characters',len(receipt),'numeric tables/figures')
