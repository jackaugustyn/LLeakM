"""Freeze the full, explicitly authorized EOS reconstruction/verification queue."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = HERE.relative_to(ROOT).as_posix()
OLD = 'experiment_validation/revalidation_20261031'


def relative(path):
    return PREFIX + '/' + path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    stages = []
    def add(name, script, args=(), checks=(), progress=None, hours=2):
        item = {'name': name, 'command': [sys.executable, relative(script), *args],
                'timeout_hours': hours,
                'success_json': [{'path': relative(path), 'equals': wanted} for path, wanted in checks]}
        if progress:
            item['progress_json'] = relative(progress)
        stages.append(item)
    add('secondary_t5', 'secondary/rerun_secondary.py',
        ['--output', relative('secondary/cuda_full'), '--device', 'cuda', '--require-complete'],
        [('secondary/cuda_full/status.json', {'complete': True, 'status': 'completed', 'completed': 3900}),
         ('secondary/cuda_full/summary.json', {'complete': True, 'n': 3900})],
        'secondary/cuda_full/status.json', 24)
    add('audit_secondary', 'validation/secondary_audit.py', checks=[('validation/secondary_audit.json', {'status': 'PASS'})])
    for label, folder, source, n in [
        ('primary', 'cuda_full', 'collection/canonical_traces.jsonl', 2100),
        ('tinyllama', 'fresh_tinyllama_cuda', 'fresh_references/tinyllama/canonical_traces.jsonl', 300),
        ('phi', 'fresh_phi_cuda', 'fresh_references/phi/canonical_traces.jsonl', 300),
    ]:
        run = 'inference/' + folder
        verification = 'inference_validation/' + folder
        add(label + '_t5', 'inference/rerun_canonical.py',
            ['--input', OLD + '/' + source, '--output', relative(run), '--device', 'cuda'],
            [(run + '/status.json', {'complete': True, 'status': 'completed', 'completed': n})],
            run + '/status.json', 24)
        add('verify_' + label, 'inference_validation/verify_inference.py',
            ['--run', relative(run), '--input', OLD + '/' + source, '--output', relative(verification)],
            [(verification + '/verification.json', {'complete': True, 'verification_status': 'VERIFIED_COMPLETE'}),
             (verification + '/statistics.json', {'complete': True})])
    add('audit_main_fresh', 'validation/audit_main_fresh.py',
        checks=[('validation/main_fresh_factledger.json', {'all_checks_passed': True})])
    add('audit_all_candidates', 'validation/audit_campaign.py',
        checks=[('validation/audit_campaign.json', {'status': 'PASS', 'complete': True,
                                                   'records_checked': 6600, 'map_predictions_matched': 3300})], hours=4)
    add('paired_rendering', 'paired_rendering.py',
        checks=[('validation/paired_detokenization_score_contrast.json', {'complete': True})])
    add('collect_complete_results', 'collect_completed.py',
        checks=[('completed_results.json', {'complete': True, 'reconstructions': 6600})])
    dependencies = [p for folder in ['inference', 'secondary', 'inference_validation', 'validation']
                    for p in (HERE / folder).glob('*') if p.suffix in {'.py', '.cpp'}]
    dependencies += [HERE / name for name in ['run_queue.py', 'paired_rendering.py', 'collect_completed.py']]
    dependencies += [ROOT / name for name in [
        'weiss_reconstruction.py', 'experiment_validation/scripts/robustness_eval.py',
        'experiment_validation/scripts/run_validation.py', 'experiment_validation/scripts/analyze_stats.py',
        'experiment_validation/scripts/generate_prompts.py',
        OLD + '/statistics/recompute_all_statistics.py',
        OLD + '/collection/canonical_traces.jsonl',
        OLD + '/fresh_references/tinyllama/canonical_traces.jsonl',
        OLD + '/fresh_references/phi/canonical_traces.jsonl',
        OLD + '/secondary/cuda_full/input_inventory.jsonl',
        OLD + '/secondary/cuda_full/calibration.json',
    ]]
    for stage in stages:
        assert (ROOT / stage['command'][1]).is_file(), stage['command'][1]
    plan = {'schema': 'full-eos-selector-campaign/1',
            'user_authorization': 'Wykonaj pełną kampanię z poprawionym selektorem',
            'purpose': 'Rerun all 6600 T5 reconstructions on frozen inputs, save every candidate, verify all metrics and MAP exact negative control.',
            'selector_version': 'eos-through-first-inclusive/1', 'expected_reconstructions': 6600,
            'monitor_interval_seconds': 15,
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(dependencies))},
            'stages': stages}
    path = HERE / 'queue_plan.json'
    raw = json.dumps(plan, indent=2, ensure_ascii=False) + '\n'
    if path.exists() and path.read_text() != raw:
        raise ValueError('A different frozen queue plan exists; do not mutate an active campaign')
    path.write_text(raw)
    print(json.dumps({'stages': [s['name'] for s in stages], 'bound_files': len(dependencies), 'reconstructions': 6600}, indent=2))


if __name__ == '__main__':
    main()
