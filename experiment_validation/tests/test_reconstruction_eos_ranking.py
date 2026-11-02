"""Selection regressions use real transition scoring and controlled token batches.

No model weights or GPU are needed. The tests exercise selection, audit output,
score-failure handling and preservation of the single-candidate sampling path.
"""
import json
from types import SimpleNamespace
import unittest

import torch
from transformers.generation.utils import GenerationMixin

import weiss_reconstruction as wr


class Tokenizer:
    eos_token_id = 1

    def __init__(self, sequences, texts):
        self.texts = {tuple(row): text for row, text in zip(sequences.tolist(), texts)}

    def __call__(self, *args, **kwargs):
        return {'input_ids': torch.tensor([[9, 1]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long)}

    def decode(self, sequence, skip_special_tokens=True):
        return self.texts[tuple(sequence.tolist())]


class TensorModel:
    """Model generation is predetermined; selection still calls real scorer API."""
    def __init__(self, sequences, transitions, eos=1):
        self.sequences = sequences
        self.transitions = transitions
        self.config = SimpleNamespace(eos_token_id=eos, vocab_size=8)
        self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.compute_calls = 0
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        scores = tuple(torch.zeros((len(self.sequences), 8)) for _ in range(self.transitions.shape[1])) if kwargs.get('output_scores') else None
        return SimpleNamespace(sequences=self.sequences, scores=scores)

    def compute_transition_scores(self, sequences, scores, normalize_logits):
        self.compute_calls += 1
        assert normalize_logits
        return self.transitions


def run_batch(sequences, transitions, texts, *, eos=1, audit=True):
    sequences = torch.tensor(sequences)
    transitions = torch.tensor(transitions, dtype=torch.float32)
    model = TensorModel(sequences, transitions, eos)
    records = [] if audit else None
    ranked = wr._sample_and_rank(model, Tokenizer(sequences, texts), 'cpu', 'lengths',
                                 samples=len(texts), candidate_audit=records)
    return ranked, records, model


class EOSRankingTests(unittest.TestCase):
    def test_real_transformers_counterexample_now_selects_short_candidate(self):
        sequences = torch.tensor([[0, 2, 1, 0, 0], [0, 3, 3, 3, 1]])
        scores = tuple(torch.tensor(x, dtype=torch.float32) for x in [
            [[-float('inf'), -3, 3, -3], [-float('inf'), 0, 0, 0]],
            [[-float('inf'), 3, -3, -3], [-float('inf'), 0, 0, 0]],
            [[-float('inf'), 0, 0, 0], [-float('inf'), 0, 0, 0]],
            [[-float('inf'), 0, 0, 0], [-float('inf'), 0, 0, 0]],
        ])

        class Model:
            config = SimpleNamespace(vocab_size=4, eos_token_id=1)
            generation_config = SimpleNamespace(eos_token_id=1)
            compute_transition_scores = GenerationMixin.compute_transition_scores

            def generate(self, **kwargs):
                return SimpleNamespace(sequences=sequences, scores=scores)

        model = Model()
        old_scores = model.compute_transition_scores(sequences, scores, normalize_logits=True).sum(1)
        self.assertEqual(old_scores.argmax().item(), 1)
        audit = []
        ranked = wr._sample_and_rank(model, Tokenizer(sequences, ['short', 'long']), 'cpu', 'test',
                                     samples=2, candidate_audit=audit)
        self.assertEqual([text for text, _ in ranked], ['short', 'long'])
        self.assertAlmostEqual(ranked[0][1], -0.00989, places=5)
        self.assertEqual(audit[0]['ranked_candidate_indices'], [0, 1])
        self.assertEqual(audit[0]['candidates'][0]['scored_token_count'], 2)
        self.assertEqual(audit[0]['candidates'][0]['legacy_full_step_confidence'], '-Infinity')
        json.dumps(audit, allow_nan=False)

    def test_no_eos_keeps_full_generated_suffix_and_excludes_decoder_prefix(self):
        ranked, audit, _ = run_batch([[0, 77, 2, 3, 4], [0, 77, 3, 4, 2]],
                                    [[-1, -2, -3], [-1, -2, -4]], ['a', 'b'])
        self.assertEqual(ranked, [('a', -6.0), ('b', -7.0)])
        self.assertEqual(audit[0]['decoder_prefix_length'], 2)
        self.assertEqual(audit[0]['candidates'][0]['generated_token_ids'], [2, 3, 4])
        self.assertIsNone(audit[0]['candidates'][0]['first_eos_generated_offset'])

    def test_first_eos_including_eos_with_pad_equals_eos_and_multiple_ids(self):
        ranked, audit, _ = run_batch([[0, 2, 0, 0, 0], [0, 3, 7, 2, 0]],
                                    [[-1, -2, -float('inf'), float('nan')], [-1, -3, -4, -5]],
                                    ['a', 'b'], eos=[0, 7])
        self.assertEqual(ranked, [('a', -3.0), ('b', -4.0)])
        self.assertEqual([c['first_eos_generated_offset'] for c in audit[0]['candidates']], [1, 1])
        self.assertEqual(audit[0]['candidates'][0]['transition_scores'][-1], 'NaN')
        json.dumps(audit, allow_nan=False)

    def test_pad_before_eos_contributes_to_the_score(self):
        ranked, audit, _ = run_batch([[0, 0, 2, 1, 0], [0, 3, 2, 1, 0]],
                                    [[-2, -3, -4, -99], [-4, -3, -4, -99]], ['a', 'b'])
        self.assertEqual(ranked[0], ('a', -9.0))
        self.assertEqual(audit[0]['candidates'][0]['scored_token_count'], 3)

    def test_eos_none_in_generation_config_does_not_use_tokenizer_eos(self):
        ranked, audit, model = run_batch([[0, 2, 1, 0], [0, 3, 1, 0]],
                                        [[-1, -2, -3], [-1, -2, -4]], ['a', 'b'], eos=None)
        self.assertEqual(ranked[0][1], -6.0)
        self.assertEqual(audit[0]['eos_token_ids'], [])
        self.assertEqual(audit[0]['candidates'][0]['scored_token_count'], 3)

    def test_nonfinite_before_or_at_eos_raises_even_for_empty_text(self):
        for position in [0, 1]:
            for invalid in [float('inf'), -float('inf'), float('nan')]:
                scores = [-1., -1., -99.]
                scores[position] = invalid
                with self.subTest(position=position, invalid=invalid), self.assertRaisesRegex(ValueError, 'before or at EOS'):
                    run_batch([[0, 2, 1, 0], [0, 3, 1, 0]], [scores, [-1, -1, -3]], ['', 'b'])

    def test_duplicate_best_candidate_and_stable_ties(self):
        ranked, audit, _ = run_batch([[0, 2, 1], [0, 3, 1], [0, 4, 1]],
                                    [[-2, -1], [-0.5, -0.5], [-0.5, -0.5]], ['same', 'other', 'same'])
        self.assertEqual(ranked, [('same', -1.0), ('other', -1.0)])
        self.assertEqual(audit[0]['ranked_candidate_indices'], [2, 1])
        ranked, audit, _ = run_batch([[0, 2, 1], [0, 3, 1]], [[-1, -1], [-1, -1]], ['same', 'same'])
        self.assertEqual(audit[0]['ranked_candidate_indices'], [0])

    def test_empty_candidates_are_retained_in_audit_but_not_ranking(self):
        ranked, audit, _ = run_batch([[0, 1, 0], [0, 1, 1]], [[-1, -99], [-2, -99]], ['', '  '])
        self.assertEqual(ranked, [])
        self.assertEqual(len(audit[0]['candidates']), 2)
        self.assertIsNone(audit[0]['selected_candidate_index'])
        self.assertTrue(all(not c['eligible'] for c in audit[0]['candidates']))

    def test_single_candidate_never_requests_or_computes_scores(self):
        ranked, audit, model = run_batch([[0, 2, 1]], [[-1, -1]], ['single'])
        self.assertEqual(ranked, [('single', 0.0)])
        self.assertNotIn('output_scores', model.kwargs)
        self.assertEqual(model.compute_calls, 0)
        self.assertIsNone(audit[0]['candidates'][0]['transition_scores'])
        self.assertIsNone(audit[0]['candidates'][0]['legacy_full_step_confidence'])
        self.assertEqual(audit[0]['candidates'][0]['scored_token_count'], 0)
        self.assertEqual(audit[0]['candidates'][0]['first_eos_generated_offset'], 1)

    def test_missing_scores_fails_instead_of_zero_ranking(self):
        seq = torch.tensor([[0, 2, 1], [0, 3, 1]])
        model = TensorModel(seq, torch.tensor([[-1., -1.], [-1., -1.]]))
        model.generate = lambda **kwargs: SimpleNamespace(sequences=seq, scores=None)
        with self.assertRaisesRegex(RuntimeError, 'requires generation scores'):
            wr._sample_and_rank(model, Tokenizer(seq, ['a', 'b']), 'cpu', 'x', samples=2)

    def test_transition_computation_failure_propagates(self):
        seq = torch.tensor([[0, 2, 1], [0, 3, 1]])
        model = TensorModel(seq, torch.tensor([[-1., -1.], [-1., -1.]]))
        def fail(*args, **kwargs):
            raise RuntimeError('broken transition scorer')
        model.compute_transition_scores = fail
        with self.assertRaisesRegex(RuntimeError, 'broken transition scorer'):
            wr._sample_and_rank(model, Tokenizer(seq, ['a', 'b']), 'cpu', 'x', samples=2)

    def test_score_sequence_dimension_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'cannot align'):
            run_batch([[0, 1], [0, 2]], [[-1, -2], [-1, -2]], ['a', 'b'])

    def test_audit_observation_preserves_rng_and_sampling_arguments(self):
        seq = torch.tensor([[0, 2, 1], [0, 3, 1]])
        class RandomModel(TensorModel):
            def generate(self, **kwargs):
                self.transitions = torch.log_softmax(torch.rand((2, 2)), dim=1)
                return super().generate(**kwargs)
        model = RandomModel(seq, torch.ones((2, 2)))
        tokenizer = Tokenizer(seq, ['a', 'b'])
        torch.manual_seed(4321)
        without = wr._sample_and_rank(model, tokenizer, 'cpu', 'x', samples=2)
        state_without = torch.get_rng_state().clone()
        kwargs_without = {k: v for k, v in model.kwargs.items() if not isinstance(v, torch.Tensor)}
        torch.manual_seed(4321)
        audit = []
        with_audit = wr._sample_and_rank(model, tokenizer, 'cpu', 'x', samples=2, candidate_audit=audit)
        self.assertEqual(without, with_audit)
        self.assertTrue(torch.equal(state_without, torch.get_rng_state()))
        self.assertEqual(kwargs_without, {k: v for k, v in model.kwargs.items() if not isinstance(v, torch.Tensor)})


if __name__ == '__main__':
    unittest.main()
