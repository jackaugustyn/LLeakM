"""Collect fully verified queue outputs into a reviewable report and manifest."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path):
    return json.loads((BASE / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    reports = {}
    for name in ['cuda_full', 'fresh_tinyllama_cuda', 'fresh_phi_cuda']:
        folder = Path('inference_validation') / name
        check, stats = read(folder / 'verification.json'), read(folder / 'statistics.json')
        if not (check['complete'] and check['verification_status'] == 'VERIFIED_COMPLETE' and stats['complete']):
            raise ValueError(f'Unverified or partial inference: {name}')
        if check['protocol_id'] != stats['protocol_id']:
            raise ValueError(f'Mixed verification/statistics protocol: {name}')
        reports[name] = {'verification': check, 'statistics': stats}
    secondary = read('secondary/cuda_full/summary.json')
    if not (secondary['complete'] and secondary['n'] == secondary['expected'] == 3900):
        raise ValueError('Secondary reconstruction design incomplete')
    for name in ['tinyllama', 'phi']:
        status = read(f'fresh_references/{name}/status.json')
        if not (status['status'] == 'completed' and status['complete'] and status['completed'] == 300):
            raise ValueError(f'Fresh generation incomplete: {name}')
    timestamp = datetime.now(timezone.utc).isoformat()
    report = {'complete': True, 'collected_utc': timestamp, 'main_reconstructions': 2100,
              'secondary_reconstructions': 3900, 'fresh_generations': 600,
              'fresh_reference_reconstructions': 600,
              'publication_integration': 'pending scientific review of complete outputs',
              'inference': reports, 'secondary': secondary}
    text = ['# Wyniki wznowionych obliczeń', '', '## Material Passport', '',
            '- Origin Skill: ARS-Codex / experiment-agent', '- Origin Mode: validate',
            f'- Origin Date: {timestamp}',
            '- Verification Status: VERIFIED — obliczenia w ramach jawnie zapisanych protokołów.',
            '- Version Label: revalidation_20261031_completed', '',
            'Ukończono i sprawdzono 2100 rekonstrukcji z archiwalnych, zweryfikowanych długości, '
            '3900 rekonstrukcji kontroli/obron/MAP oraz 600 nowych generacji ofiar i ich 600 rekonstrukcji T5. '
            'Nowe generacje mają własne wersje checkpointów i zweryfikowane dekodowanie całej sekwencji; '
            'nie są odzyskanymi historycznymi odpowiedziami.', '',
            '| Seria | Model | N | Średnia φ |', '|---|---|---:|---:|']
    for name, content in reports.items():
        for model in content['statistics']['models']:
            text.append(f"| {name} | {model['label']} | {model['n']} | {model['metrics']['phi']['mean']:.6f} |")
    text += ['', 'Pełne metryki, przedziały i testy par znajdują się w odpowiednich katalogach '
             '`inference_validation/*/statistics.json`. Wyniki pilotaży i MAP: '
             '`secondary/cuda_full/summary.json`.', '',
             'Ważne granice interpretacji: dwie archiwalne serie nadal mają wadliwe referencje; '
             'nowe referencje są osobnym eksperymentem. Zmiana kolektora, detokenizacji, środowiska '
             'i wersji wag nie została rozdzielona przyczynowo. Limit 80 tokenów enkodera nadal '
             'ogranicza obserwację. Metryki leksykalne nie mierzą bezpośrednio ujawnienia danych.', '',
             'Artykuł wymaga teraz uzgodnienia z kompletnymi wynikami. Kolejka nie zmienia automatycznie '
             'wniosków ani nie łączy nowych odpowiedzi z historycznym rankingiem.', '']
    (BASE / 'WYNIKI_WZNOWIENIA.md').write_text('\n'.join(text))
    (BASE / 'completed_results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    files = [p for p in BASE.rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.jsonl', '.md'}
             and '__pycache__' not in p.parts and 'queue' not in p.parts
             and p.name != 'completed_manifest.json']
    files += [ROOT / p for p in ['app.py', 'weiss_reconstruction.py', 'article1/template.tex', 'article1/template.pdf']]
    manifest = {'captured_utc': timestamp, 'scope': 'Current completed rerun artifacts and paper state; no historical attestation',
                'files': {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(set(files))}}
    (BASE / 'completed_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'complete': True, 'report': str(BASE / 'WYNIKI_WZNOWIENIA.md')}))


if __name__ == '__main__':
    main()
