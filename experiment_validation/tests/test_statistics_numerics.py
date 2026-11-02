"""Regression checks for scientific calculation failures; no model inference."""
from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment_validation.scripts import analyze_stats, defense_eval, trace_baselines


class SignedRankNumericsTests(unittest.TestCase):
    def test_extreme_tail_does_not_round_to_zero(self):
        n = 300
        values = np.arange(1, n + 1, dtype=float)
        expected_z = (-n * (n + 1) / 4 + .5) / math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        expected_p = math.erfc(abs(expected_z) / math.sqrt(2))
        for fn in (analyze_stats.wilcoxon_signed_rank, defense_eval.wilcoxon_signed_rank):
            result = fn(values, np.zeros(n))
            self.assertGreater(result['p_value'], 0)
            self.assertAlmostEqual(result['p_value'] / expected_p, 1.0, places=13)
            self.assertEqual(result['rank_biserial'], 1.0)

    def test_balanced_signed_ranks_have_p_one(self):
        # All absolute differences tie, and the positive and negative rank sums agree.
        values = np.array([-1., 1.] * 30)
        for fn in (analyze_stats.wilcoxon_signed_rank, defense_eval.wilcoxon_signed_rank):
            result = fn(values, np.zeros(len(values)))
            self.assertEqual(result['p_value'], 1.0)
            self.assertEqual(result['rank_biserial'], 0.0)

    def test_zero_differences_are_removed(self):
        a = np.array([0., 0., 1., -2., 4., -5.])
        for fn in (analyze_stats.wilcoxon_signed_rank, defense_eval.wilcoxon_signed_rank):
            full = fn(a, np.zeros(len(a)))
            trimmed = fn(a[2:], np.zeros(len(a) - 2))
            self.assertEqual(full, trimmed)


class TraceAggregationTests(unittest.TestCase):
    def test_resume_aggregates_both_percentiles_without_inference(self):
        rows = [{'idx': i, 'topic': 'topic', 'run_id': str(i), 'response_text': 'reference'} for i in (1, 2)]
        conditions = ('real', 'shuffled', 'constant', 'cross_prompt')
        saved = [dict(label='fixture', model='Fixture', idx=i, topic='topic', condition=c,
                      phi=v, pred='saved') for c in conditions for i, v in [(1, .1), (2, .5)]]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'trace_baseline_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in saved))
            with patch.object(trace_baselines, 'OUT', folder), \
                 patch.object(trace_baselines, 'RUNS', [('fixture', 'Fixture', 'unused')]), \
                 patch.object(trace_baselines, 'load_samples', return_value=rows), \
                 patch.object(trace_baselines, 'token_lengths_for', return_value=[1, 2, 3]), \
                 patch.dict(sys.modules, {'torch': types.SimpleNamespace()}), \
                 patch('weiss_reconstruction.reconstruct', side_effect=AssertionError('Unexpected inference')), \
                 patch.object(sys, 'argv', ['trace_baselines.py']), contextlib.redirect_stdout(io.StringIO()):
                trace_baselines.main()
            summary = json.loads((folder / 'trace_baselines.json').read_text())
            self.assertEqual(len(summary), 4)
            for item in summary:
                self.assertEqual(item['n'], 2)
                self.assertAlmostEqual(item['phi_mean'], .3)
                self.assertEqual(item['phi_ci'], [.1, .5])


if __name__ == '__main__':
    unittest.main()
