"""Check manuscript tables/figures against independently audited fact ledgers."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TEX=ROOT/'article1/template.tex'
NAMES={'Qwen2.5-1.5B':'qwen_1_5b','Qwen2.5-3B':'qwen_3b','Phi-1.5':'phi_1_5',
       'Llama-3.2-3B':'llama_3_2_3b','Gemma-2-2B':'gemma_2_2b',
       'TinyLlama-1.1B':'tinyllama_1_1b','Phi-3.5-mini':'phi_3_5_mini'}
NUMBER=re.compile(r'[+-]?\d+(?:\.\d+)?')


def main():
    text=TEX.read_text();ledger=json.loads((HERE/'main_fresh_factledger.json').read_text())
    secondary=json.loads((HERE/'secondary_audit.json').read_text())
    primary={r['label']:r for r in ledger['cohorts']['cuda_full']['models']}
    fresh={r['label']:r for name,c in ledger['cohorts'].items() if name!='cuda_full' for r in c['models']}
    cells={(r['suite'],r['label'],r['condition'],r['attacker']):r for r in secondary['cells']}
    contrasts={(r['suite'],r['label'],r['condition'],r['attacker']):r for r in secondary['paired_contrasts']}
    tables={}
    for block in re.findall(r'\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}',text):
        label=re.search(r'\\label\{([^}]+)\}',block)
        if label:tables[label[1]]=block
    checks=[];errors=[];rowcounts={}
    def compare(where,values,expected):
        actual=NUMBER.findall(values);expected=[x for x in expected if x is not None]
        if len(actual)!=len(expected):
            errors.append({'where':where,'actual_numbers':actual,'expected':expected,'error':'numeric field count'});return
        for token,value in zip(actual,expected):
            digits=len(token.split('.')[1]) if '.' in token else 0
            ok=abs(float(token)-value)<=.500000001*10**(-digits)+1e-12
            checks.append({'where':where,'printed':token,'expected':value,'pass':ok})
            if not ok:errors.append(checks[-1])
    def rows(label):
        block=tables[label];body=block[block.index('\\midrule')+len('\\midrule'):];answer=[]
        for line in body.splitlines():
            if '&' in line and '\\\\' in line and not line.lstrip().startswith('\\'):
                answer.append([s.strip() for s in line.split('&')])
        rowcounts[label]=len(answer)
        return answer
    primary_tables=['tab:main_results','tab:statistical_results','tab:scenario_sensitivity','tab:eos_completeness','tab:length_covariates']
    for label in primary_tables+['tab:fresh_results']:
        for cols in rows(label):
            lab=next(v for k,v in NAMES.items() if cols[0].startswith(k));r=(fresh if label=='tab:fresh_results' else primary)[lab]
            phi=r['metrics']['phi']['mean'];mp=r['metrics'];start=1
            if label=='tab:main_results':
                start=2;expected=[phi,mp['ed_norm']['mean'],mp['rouge1_precision']['mean'],mp['rougeL_precision']['mean'],r['LR_percent']]
            elif label=='tab:statistical_results':expected=[phi,*mp['phi']['ci_prompt_bootstrap'],r['LR_percent'],*r['LR_wilson_percent']]
            elif label=='tab:scenario_sensitivity':expected=[phi,*r['phi_ci_scenario_bootstrap']]
            elif label=='tab:eos_completeness':expected=[r['EOS_count'],r['EOS_percent'],r['token_limit_percent'],phi,r['EOS_only_phi_mean'],r['LR_percent'],r['EOS_only_LR_percent']]
            elif label=='tab:length_covariates':expected=[r['tokens_mean'],r['reference_characters_mean'],r['token_bytes_mean_of_sample_means'],r['token_limit_percent']]
            else:expected=[phi,*r['phi_ci_scenario_bootstrap'],mp['ed_norm']['mean'],mp['rouge1_precision']['mean'],mp['rougeL_precision']['mean'],r['LR_percent'],r['EOS_percent']]
            compare(label+'/'+lab,' '.join(cols[start:]),expected)
    suites={'tab:trace_controls':'trace_control','tab:defense_results':'defense_pilot','tab:robustness':'map'}
    for label,suite in suites.items():
        lab=None
        for cols in rows(label):
            if cols[0]:lab=next(v for k,v in NAMES.items() if cols[0].startswith(k))
            condition=cols[1].replace('\\_','_').replace('cross-prompt','cross_prompt').replace('true','real')
            attacker=cols[2] if suite=='map' else 'frozen';key=suite,lab,condition,attacker;r=cells[key]
            baseline=condition in {'baseline','real'}
            expected=[r['metric_means']['phi'],*r['phi_ci']]
            if suite=='trace_control':expected+=[None if baseline else contrasts[key]['mean_phi_difference_vs_baseline']];start=2
            elif suite=='defense_pilot':expected+=[None if baseline else contrasts[key]['score_reduction_pct'],r['observed_payload_overhead_pct_mean'],r['observed_units_mean']];start=2
            else:expected+=[r['between_seed_sd'],0 if baseline else contrasts[key]['score_reduction_pct']];start=3
            compare(label+'/'+lab+'/'+condition+'/'+attacker,' '.join(cols[start:]),expected)
    for cols in rows('tab:encoder_coverage'):
        condition=cols[0].replace('\\_','_')
        expected=[]
        for lab in ['qwen_1_5b','llama_3_2_3b']:expected += [cells['map',lab,condition,'frozen']['unique_first_inputs_truncated'],30]
        compare('tab:encoder_coverage/'+condition,' '.join(cols[1:]),expected)
    figures={}
    for block in re.findall(r'\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}',text):
        label=re.search(r'\\label\{([^}]+)\}',block)
        if label:figures[label[1]]=block
    chart=figures['fig:model_ci'];data=re.search(r'table\[[^\]]+\]\s*\{([\s\S]*?)\};',chart)[1]
    codes={'Qwen1.5':'qwen_1_5b','Qwen3':'qwen_3b','Phi1.5':'phi_1_5','Llama3':'llama_3_2_3b','Gemma2':'gemma_2_2b','Phi3.5':'phi_3_5_mini','TinyLlama':'tinyllama_1_1b'}
    for line in data.splitlines():
        parts=line.split()
        if parts and parts[0] in codes:
            lab=codes[parts[0]];r=primary[lab]['metrics']['phi'];mean=r['mean'];lo,hi=r['ci_prompt_bootstrap']
            compare('fig:model_ci/'+lab,' '.join(parts[1:]),[mean,hi-mean,mean-lo])
    coords=re.search(r'coordinates\s*\{([^}]+)\}',figures['fig:defense_tradeoff'])[1]
    expected=[n for r in secondary['defense_figure_equal_model_average'] for n in [r['payload_overhead_pct'],r['mean_model_score_reduction_pct']]]
    compare('fig:defense_tradeoff',coords,expected)
    expectedcounts={'tab:main_results':7,'tab:statistical_results':7,'tab:scenario_sensitivity':7,'tab:eos_completeness':7,'tab:length_covariates':7,'tab:fresh_results':2,'tab:trace_controls':8,'tab:defense_results':12,'tab:robustness':22,'tab:encoder_coverage':6}
    for table,n in expectedcounts.items():
        if rowcounts[table]!=n:errors.append({'table':table,'expected_rows':n,'actual_rows':rowcounts[table]})
    report={'status':'PASS' if not errors else 'FAIL','manuscript_sha256':hashlib.sha256(TEX.read_bytes()).hexdigest(),
            'numeric_fields_checked':len(checks),'row_counts':rowcounts,'figures_checked':2,'errors':errors,
            'checks':checks,'scope':'Actual TeX table cells and figure coordinates checked against independently audited result ledgers; does not by itself certify prose or rendered layout.'}
    (HERE/'manuscript_numeric_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
