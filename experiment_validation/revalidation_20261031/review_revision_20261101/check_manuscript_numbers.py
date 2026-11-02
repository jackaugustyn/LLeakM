"""Check manuscript tables/figures against independently audited fact ledgers."""
from pathlib import Path
import hashlib
import ast
from collections import Counter, defaultdict
import json
import re
import argparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TEX=ROOT/'article1/template.tex'
FACTS=HERE.parent/'final_review_20261101'
FIGURES=HERE.parent/'article_revision_latest_20261101'
NAMES={'Qwen2.5-1.5B':'qwen_1_5b','Qwen2.5-3B':'qwen_3b','Phi-1.5':'phi_1_5',
       'Llama-3.2-3B':'llama_3_2_3b','Gemma-2-2B':'gemma_2_2b',
       'TinyLlama-1.1B':'tinyllama_1_1b','Phi-3.5-mini':'phi_3_5_mini'}
NUMBER=re.compile(r'[+-]?\d+(?:\.\d+)?')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables-only',action='store_true',help='Interim table check; omits figures and writes a separate report.')
    args=parser.parse_args()
    artifact_hashes={}
    text=TEX.read_text();ledger=json.loads((FACTS/'main_fresh_factledger.json').read_text())
    secondary=json.loads((FACTS/'secondary_audit.json').read_text())
    primary={r['label']:r for r in ledger['cohorts']['cuda_full']['models']}
    fresh={r['label']:r for name,c in ledger['cohorts'].items() if name!='cuda_full' for r in c['models']}
    cells={(r['suite'],r['label'],r['condition'],r['attacker']):r for r in secondary['cells']}
    contrasts={(r['suite'],r['label'],r['condition'],r['attacker']):r for r in secondary['paired_contrasts']}
    tables={}
    for block in re.findall(r'\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}',text):
        label=re.search(r'\\label\{([^}]+)\}',block)
        if label:tables[label[1]]=block
    checks=[];errors=[];rowcounts={};text_checks=[];figure_checks=[]
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
    # Bind the two descriptive tables to retained local metadata and all source records.
    # Parameter/tokenizer family strings are consistency checks, not new model-card certification.
    meta_file=ROOT/'experiment_validation/scripts/analyze_extras.py'
    tree=ast.parse(meta_file.read_text())
    metadata={r[0]:r for node in tree.body if isinstance(node,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='RUNS' for t in node.targets)
              for r in ast.literal_eval(node.value)}
    model_ids={r['label'].removesuffix('_full'):r['model_id']
               for r in json.loads((ROOT/'experiment_validation/models_publication.json').read_text())}
    source_records=[json.loads(line) for line in (HERE.parent/'collection/canonical_traces.jsonl').read_text().splitlines() if line]
    source_dirs=defaultdict(set);source_counts=Counter()
    for r in source_records:
        source_dirs[r['label']].add(r['run_dir']);source_counts[r['label']]+=1
    def plain(value):
        return re.sub(r'\\[A-Za-z]+', '', value).replace('\\_', '_').replace('{','').replace('}','').replace('\\','').strip()
    def text_equal(where,actual,expected):
        ok=actual==expected;item={'where':where,'actual':actual,'expected':expected,'pass':ok}
        text_checks.append(item)
        if not ok:errors.append(item)
    for cols in rows('tab:model_metadata'):
        lab=next(v for k,v in NAMES.items() if cols[0].startswith(k));r=metadata[lab]
        text_equal('tab:model_metadata/'+lab+'/column_count',len(cols),4)
        values=[plain(c) for c in cols]
        for field,actual,expected in zip(['model','checkpoint','parameters','tokenizer'],values,[r[1],model_ids[lab],r[5],r[6]]):
            text_equal('tab:model_metadata/'+lab+'/'+field,actual,expected)
    for cols in rows('tab:run_ids'):
        lab=next(v for k,v in NAMES.items() if cols[0].startswith(k))
        text_equal('tab:run_ids/'+lab+'/column_count',len(cols),2)
        text_equal('tab:run_ids/'+lab+'/model',plain(cols[0]),metadata[lab][1])
        text_equal('tab:run_ids/'+lab+'/directory',[plain(cols[1])],sorted(source_dirs[lab]))
        text_equal('tab:run_ids/'+lab+'/record_count',source_counts[lab],300)
        text_equal('tab:run_ids/'+lab+'/directory_exists',(ROOT/'experiment_validation/results'/plain(cols[1])).is_dir(),True)

    if not args.tables_only:
        # Figures are generated from these exact inputs; numeric provenance is checked separately
        # from their rendering. The final report hashes the generator, data and exported artifacts.
        figures={}
        for block in re.findall(r'\\begin\{figure\*?\}[\s\S]*?\\end\{figure\*?\}',text):
            label=re.search(r'\\label\{([^}]+)\}',block)
            if label:figures[label[1]]=block
        text_equal('manuscript/complete_figure_set',sorted(figures),sorted(['fig:threat_model','fig:pipeline','fig:model_ci','fig:defense_tradeoff']))
        data_path=FIGURES/'figure_data.json'
        figure_data=json.loads(data_path.read_text())
        def model_label(value):
            if value in primary:return value
            return next(v for k,v in NAMES.items() if value.startswith(k))
        for series,dataset in [('archived_scenario',primary),('fresh_scenario',fresh)]:
            records=figure_data[series]
            text_equal('fig:model_ci/'+series+'/models',sorted(model_label(r['model']) for r in records),sorted(dataset))
            for r in records:
                lab=model_label(r['model']);expected=dataset[lab]
                for field,value in zip(['mean','low','high'],[expected['metrics']['phi']['mean'],*expected['phi_ci_scenario_bootstrap']]):
                    ok=abs(r[field]-value)<1e-12;item={'where':'fig:model_ci/'+series+'/'+lab+'/'+field,'actual':r[field],'expected':value,'pass':ok}
                    figure_checks.append(item)
                    if not ok:errors.append(item)
        for series,lab in [('defense_qwen','qwen_1_5b'),('defense_llama','llama_3_2_3b')]:
            expected_conditions=sorted(k[2] for k in cells if k[0]=='defense_pilot' and k[1]==lab and k[2]!='baseline')
            text_equal('fig:defense_tradeoff/'+series+'/conditions',sorted(r['condition'] for r in figure_data[series]),expected_conditions)
            for r in figure_data[series]:
                key='defense_pilot',lab,r['condition'],'frozen'
                for field,value in [('overhead',cells[key]['observed_payload_overhead_pct_mean']),('reduction',contrasts[key]['score_reduction_pct'])]:
                    ok=abs(r[field]-value)<1e-12;item={'where':'fig:defense_tradeoff/'+lab+'/'+r['condition']+'/'+field,'actual':r[field],'expected':value,'pass':ok}
                    figure_checks.append(item)
                    if not ok:errors.append(item)
        paired=json.loads((FACTS/'paired_detokenization_score_contrast.json').read_text())
        # The source structure is handled below once its model rows are identified.
        paired_rows=paired.get('models',paired.get('contrasts',{}))
        if isinstance(paired_rows,list):paired_rows={model_label(r.get('label',r.get('model'))):r for r in paired_rows}
        for r in figure_data.get('paired_contrasts',[]):
            lab=model_label(r['model']);p=paired_rows[lab]
            mean=p.get('mean_difference',p.get('mean_phi_difference',p.get('delta_phi_mean')))
            ci=p.get('delta_phi_ci_paired_scenario',p.get('ci_scenario_bootstrap',p.get('paired_scenario_ci',p.get('paired_scenario_bootstrap_95ci'))))
            if mean is None or ci is None:raise ValueError('Unrecognized paired source row: '+str(p.keys()))
            for field,value in zip(['mean_difference','low','high'],[mean,*ci]):
                ok=abs(r[field]-value)<1e-12;item={'where':'fig:model_ci/paired/'+lab+'/'+field,'actual':r[field],'expected':value,'pass':ok}
                figure_checks.append(item)
                if not ok:errors.append(item)
        for label,stem in [('fig:model_ci','latest_model_results'),('fig:defense_tradeoff','latest_defense_tradeoff')]:
            text_equal(label+'/includegraphics',f'figures/{stem}.pdf' in figures[label],True)
        for label in ['fig:threat_model','fig:pipeline']:
            text_equal(label+'/exists',label in figures,True)
            text_equal(label+'/TikZ_source',r'\begin{tikzpicture}' in figures[label],True)
        artifacts=[TEX,data_path,FIGURES/'generate_figures.py',meta_file,ROOT/'experiment_validation/models_publication.json',
                   FACTS/'main_fresh_factledger.json',FACTS/'secondary_audit.json',FACTS/'paired_detokenization_score_contrast.json']
        for stem in ['latest_model_results','latest_defense_tradeoff']:
            for ext in ['pdf','svg','png']:
                path=ROOT/'article1/figures'/f'{stem}.{ext}';artifacts.append(path)
                text_equal('figure_artifact/'+path.name+'/exists',path.is_file(),True)
        artifact_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts if path.is_file()}
        manifest_path=FIGURES/'figure_manifest.json'
        manifest=json.loads(manifest_path.read_text())
        for group in ['sources','outputs']:
            for path,expected_hash in manifest[group].items():
                actual_hash=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
                text_equal('figure_manifest/'+group+'/'+path,actual_hash,expected_hash)
        text_equal('figure_manifest/generator_sha256',hashlib.sha256((FIGURES/'generate_figures.py').read_bytes()).hexdigest(),manifest['generator_sha256'])
        text_equal('figure_manifest/data_sha256',hashlib.sha256(data_path.read_bytes()).hexdigest(),manifest['data_sha256'])
        artifact_hashes[str(manifest_path.relative_to(ROOT))]=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    protocol_block=tables['tab:protocol_overview']
    protocol_body=protocol_block.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    protocol_rows=[[x.strip() for x in row.split('&')] for row in protocol_body.split(r'\\') if '&' in row]
    protocol_sources=[('Archived main',len(source_records),512,'All / 3'),
                      ('Verified rendering',sum(f['n'] for k,f in ledger['cohorts'].items() if k!='cuda_full'),512,'All / 3'),
                      ('Trace controls',sum(c['n'] for c in secondary['cells'] if c['suite']=='trace_control'),96,r'$\leq3$ / 3'),
                      ('Defense pilot',sum(c['n'] for c in secondary['cells'] if c['suite']=='defense_pilot'),96,r'$\leq3$ / 3'),
                      ('MAP study',sum(c['n'] for c in secondary['cells'] if c['suite']=='map'),512,'First / 1')]
    rowcounts['tab:protocol_overview']=len(protocol_rows)
    text_equal('tab:protocol_overview/design_count',len(protocol_rows),len(protocol_sources))
    for cols,(name,n,budget,scope) in zip(protocol_rows,protocol_sources):
        text_equal('tab:protocol_overview/'+name+'/name',cols[0].startswith(name),True)
        compare('tab:protocol_overview/'+name+'/N',cols[2],[n])
        compare('tab:protocol_overview/'+name+'/victim_budget',cols[3],[budget])
        text_equal('tab:protocol_overview/'+name+'/scope',cols[4],scope)
    expectedcounts={'tab:protocol_overview':5,'tab:main_results':7,'tab:statistical_results':7,'tab:scenario_sensitivity':7,'tab:eos_completeness':7,'tab:length_covariates':7,'tab:fresh_results':2,'tab:trace_controls':8,'tab:defense_results':12,'tab:robustness':22,'tab:encoder_coverage':6,'tab:model_metadata':7,'tab:run_ids':7}
    text_equal('manuscript/complete_table_set',sorted(tables),sorted(expectedcounts))
    for table,n in expectedcounts.items():
        if rowcounts[table]!=n:errors.append({'table':table,'expected_rows':n,'actual_rows':rowcounts[table]})
    for label in primary_tables+['tab:fresh_results','tab:model_metadata','tab:run_ids']:
        model_keys=[next(v for k,v in NAMES.items() if cols[0].startswith(k)) for cols in rows(label)]
        text_equal(label+'/complete_unique_model_set',sorted(model_keys),sorted(fresh if label=='tab:fresh_results' else primary))
    for label,suite in suites.items():
        actual_keys=[];lab=None
        for cols in rows(label):
            if cols[0]:lab=next(v for k,v in NAMES.items() if cols[0].startswith(k))
            condition=cols[1].replace('\\_','_').replace('cross-prompt','cross_prompt').replace('true','real')
            attacker=cols[2] if suite=='map' else 'frozen'
            actual_keys.append((suite,lab,condition,attacker))
        text_equal(label+'/complete_unique_condition_set',sorted(actual_keys),sorted(k for k in cells if k[0]==suite))
    report={'status':'PASS' if not errors else 'FAIL','manuscript_sha256':hashlib.sha256(TEX.read_bytes()).hexdigest(),
            'numeric_fields_checked':len(checks)+len(figure_checks),'table_numeric_fields_checked':len(checks),
            'figure_numeric_fields_checked':len(figure_checks),'text_fields_checked':len(text_checks),
            'row_counts':rowcounts,'figures_checked':{'numeric':0 if args.tables_only else 2,'conceptual_presence_only':0 if args.tables_only else 2},'errors':errors,
            'artifact_sha256':artifact_hashes,'figure_checks':figure_checks,'text_checks':text_checks,
            'tables_only':args.tables_only,'checks':checks,'scope':('Interim run: all 13 TeX tables checked; figures omitted. ' if args.tables_only else 'All 13 TeX tables and 2 numeric figure data files checked. ')+'Sources are independent result ledgers and retained source inventories. Metadata parameter/tokenizer descriptions checked for local consistency, not new model-card certification. Conceptual diagrams are checked only for presence in the full run; separate scientific review and visual inspection remain required. Does not certify prose or rendered layout.'}
    (HERE/('table_numeric_validation.json' if args.tables_only else 'manuscript_numeric_validation.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'checks','figure_checks','text_checks'}},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
