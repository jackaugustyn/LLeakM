"""Promote only the complete, independently checked EOS campaign to final results."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    counts = {'inference/cuda_full': 2100, 'inference/fresh_tinyllama_cuda': 300,
              'inference/fresh_phi_cuda': 300, 'secondary/cuda_full': 3900}
    protocols = {}
    for run, n in counts.items():
        status = read(run + '/status.json')
        assert status['complete'] and status['status'] == 'completed'
        assert status['completed'] == status['expected'] == n
        protocols[run] = status['protocol_id']
    for cohort in ['cuda_full', 'fresh_tinyllama_cuda', 'fresh_phi_cuda']:
        check = read('inference_validation/' + cohort + '/verification.json')
        assert check['complete'] and check['verification_status'] == 'VERIFIED_COMPLETE'
    main_facts = read('validation/main_fresh_factledger.json')
    secondary = read('validation/secondary_audit.json')
    candidates = read('validation/audit_campaign.json')
    rendering = read('validation/paired_detokenization_score_contrast.json')
    assert main_facts['all_checks_passed'] and secondary['status'] == 'PASS'
    assert candidates['status'] == 'PASS' and candidates['complete']
    assert candidates['records_checked'] == 6600 and candidates['map_predictions_matched'] == 3300
    assert rendering['complete']
    plan = read('queue_plan.json')
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    output = {'complete': True, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'selector_version': 'eos-through-first-inclusive/1', 'reconstructions': 6600,
              'fresh_victim_generations': 0, 'reused_verified_victim_responses': 600,
              'protocol_ids': protocols, 'scope': 'Complete corrected-selector reconstructions on unchanged references and length inputs; no semantic or transport-extraction claim.',
              'cohorts': main_facts['cohorts'], 'secondary': secondary,
              'selector_audit': candidates, 'paired_rendering': rendering,
              'source_sha256': plan['source_sha256']}
    (HERE / 'completed_results.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    previous = json.loads((ROOT / 'experiment_validation/revalidation_20261031/final_review_20261101/main_fresh_factledger.json').read_text())
    report = ['# Pełna kampania z poprawionym selektorem — ukończona', '',
              'Wykonano i zweryfikowano **6600/6600** nowych rekonstrukcji. Wszystkie **3300/3300** jednokandydatowych predykcji MAP są identyczne z poprzednią kampanią.', '',
              'Zmiana dotyczy sumowania wyników do pierwszego EOS włącznie; referencje, długości i seedy zostały zachowane. Wszystkich kandydatów i ich wyniki zapisano dla każdego segmentu.', '',
              '| Kohorta | Model | Poprzednia średnia φ | Nowa średnia φ | Zmiana |',
              '|---|---|---:|---:|---:|']
    for cohort, data in main_facts['cohorts'].items():
        old = {r['label']: r for r in previous['cohorts'][cohort]['models']}
        for r in data['models']:
            a = old[r['label']]['metrics']['phi']['mean']; b = r['metrics']['phi']['mean']
            report.append(f"| {cohort} | {r['model']} | {a:.6f} | {b:.6f} | {b-a:+.6f} |")
    report += ['', 'Archiwalne TinyLlama/Phi-3.5 nadal są diagnostyką referencji; oddzielnych świeżych kohort nie scala się z pozostałymi modelami w nowy ranking.', '',
               f"Kontrole: **{secondary['significant_control_contrasts']}/6** kontrastów istotnych po Holm. Obrony: **{secondary['significant_defense_contrasts']}/10**.", '',
               'Szczegółowe porównania sparowane, przedziały ufności i audyt wyboru kandydatów: `validation/audit_campaign.json`. Wszystkie liczby, metryki i protokoły: `completed_results.json`.', '',
               'Ten raport opisuje ukończone obliczenia. Artykuł wymaga integracji tych wyników i kontroli nowego PDF; wcześniejszy artykuł opisuje poprzednią kampanię.']
    (HERE / 'WYNIKI.md').write_text('\n'.join(report) + '\n')
    files = [p for p in HERE.rglob('*') if p.is_file() and p.name not in {'completed_manifest.json', 'status.json', 'supervisor.lock'}
             and not any(x in p.relative_to(HERE).parts for x in ['queue', '__pycache__', 'before', 'preflight'])
             and p.suffix not in {'.so', '.tmp'}]
    files += [ROOT / name for name in plan['source_sha256']]
    inventory = {'created_utc': output['completed_utc'], 'scope': 'Completed EOS campaign, fixed code and verified outputs; excludes live queue and preflight.',
                 'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(files))]}
    (HERE / 'completed_manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'complete': True, 'reconstructions': 6600, 'map_predictions_matched': 3300,
                      'manifest_files': len(inventory['files'])}, indent=2))


if __name__ == '__main__':
    main()
