"""Recalculate the matched rendering contrast for the complete EOS campaign."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    paths = [HERE / 'inference_validation' / name / 'rescored.jsonl'
             for name in ['cuda_full', 'fresh_tinyllama_cuda', 'fresh_phi_cuda']]
    for path in paths:
        check = json.loads((path.parent / 'verification.json').read_text())
        assert check['complete'] and check['verification_status'] == 'VERIFIED_COMPLETE'
    archived = {(r['label'], r['idx']): r for r in rows(paths[0])}
    rng = np.random.default_rng(20261101)
    result = {'complete': True, 'selector_version': 'eos-through-first-inclusive/1',
              'scope': 'Matched full-response contrast of verified versus legacy rendering under the corrected selector. Both length inputs and scoring references change; the components are not separately identified.',
              'bootstrap_seed': 20261101, 'bootstrap_draws': 10000,
              'order': 'Tiny prompt/scenario then Phi prompt/scenario', 'models': [],
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    for path, label in zip(paths[1:], ['tinyllama_1_1b', 'phi_3_5_mini']):
        fresh = sorted(rows(path), key=lambda r: r['idx'])
        assert len(fresh) == 300 and len({r['idx'] for r in fresh}) == 300
        differences = []
        for row in fresh:
            old = archived[label, row['idx']]
            assert row['scenario'] == old['scenario']
            differences.append(row['metrics']['phi'] - old['metrics']['phi'])
        values = np.array(differences)
        scenarios = sorted({r['scenario'] for r in fresh})
        assert len(scenarios) == 60 and set(Counter(r['scenario'] for r in fresh).values()) == {5}
        means = np.array([np.mean([d for r, d in zip(fresh, differences) if r['scenario'] == scenario]) for scenario in scenarios])
        intervals = []
        for series in [values, means]:
            draws = rng.integers(0, len(series), size=(10000, len(series)))
            intervals.append(np.percentile(series[draws].mean(axis=1), [2.5, 97.5]).tolist())
        result['models'].append({'label': label, 'n': 300, 'n_scenarios': 60,
                                 'delta_phi_mean': float(values.mean()),
                                 'delta_phi_ci_paired_prompt': intervals[0],
                                 'delta_phi_ci_paired_scenario': intervals[1],
                                 'individual_delta_range': [float(values.min()), float(values.max())]})
    independent_path = HERE / 'validation/paired_rendering_audit.json'
    independent = json.loads(independent_path.read_text())
    for calculated, checked in zip(result['models'], independent['models']):
        assert calculated['label'] == checked['label']
        for key in ['delta_phi_mean', 'delta_phi_ci_paired_prompt', 'delta_phi_ci_paired_scenario']:
            assert np.allclose(calculated[key], checked[key], rtol=0, atol=1e-12), key
    result['independent_recalculation_agrees'] = True
    result['source_sha256'][str(independent_path.relative_to(ROOT))] = hashlib.sha256(independent_path.read_bytes()).hexdigest()
    out = HERE / 'validation/paired_detokenization_score_contrast.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
