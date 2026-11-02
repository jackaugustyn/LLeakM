import unittest
from types import SimpleNamespace
import torch
from transformers.generation.utils import GenerationMixin
from eos_scoring import scores_through_eos


class EOSScoringTests(unittest.TestCase):
    def test_reviewer_counterexample_actual_transformers_scores(self):
        class FakeModel:
            config = SimpleNamespace(vocab_size=4)
            compute_transition_scores = GenerationMixin.compute_transition_scores
        seq = torch.tensor([[0, 2, 1, 0, 0], [0, 3, 3, 3, 1]])
        scores = tuple(torch.tensor(x, dtype=torch.float32) for x in [
            [[-float('inf'), -3, 3, -3], [-float('inf'), 0, 0, 0]],
            [[-float('inf'), 3, -3, -3], [-float('inf'), 0, 0, 0]],
            [[-float('inf'), 0, 0, 0], [-float('inf'), 0, 0, 0]],
            [[-float('inf'), 0, 0, 0], [-float('inf'), 0, 0, 0]],
        ])
        transitions = FakeModel().compute_transition_scores(seq, scores, normalize_logits=True)
        fixed, counts, offsets = scores_through_eos(seq, transitions, 1)
        self.assertEqual(transitions.sum(1).argmax().item(), 1)
        self.assertEqual(fixed.argmax().item(), 0)
        self.assertEqual(counts, [2, 4])
        self.assertEqual(offsets, [1, 3])
        self.assertAlmostEqual(fixed[0].item(), -0.00989, places=5)

    def test_no_eos_retains_full_generated_suffix(self):
        seq = torch.tensor([[9, 9, 2, 3, 4]])
        fixed, counts, offsets = scores_through_eos(seq, torch.tensor([[-1., -2., -3.]]), 1)
        self.assertEqual(fixed.tolist(), [-6.])
        self.assertEqual(counts, [3])
        self.assertEqual(offsets, [None])

    def test_first_of_multiple_eos_and_pad_equals_eos(self):
        seq = torch.tensor([[0, 2, 0, 0, 0], [0, 3, 7, 2, 0]])
        transitions = torch.tensor([[-1., -2., -float('inf'), -float('inf')], [-1., -3., -4., -5.]])
        fixed, counts, offsets = scores_through_eos(seq, transitions, [0, 7])
        self.assertEqual(fixed.tolist(), [-3., -4.])
        self.assertEqual(counts, [2, 2])

    def test_pad_before_eos_is_not_removed(self):
        seq = torch.tensor([[0, 0, 2, 1, 0]])
        fixed, counts, _ = scores_through_eos(seq, torch.tensor([[-2., -3., -4., -99.]]), 1)
        self.assertEqual(fixed.tolist(), [-9.])
        self.assertEqual(counts, [3])

    def test_in_sequence_nonfinite_is_not_hidden(self):
        seq = torch.tensor([[0, 2, 1, 0]])
        fixed, _, _ = scores_through_eos(seq, torch.tensor([[-float('inf'), -1., -4.]]), 1)
        self.assertTrue(torch.isneginf(fixed[0]).item())

    def test_shape_failure(self):
        with self.assertRaises(ValueError):
            scores_through_eos(torch.tensor([[0, 1]]), torch.ones(1, 3), 1)


if __name__ == '__main__':
    unittest.main()
