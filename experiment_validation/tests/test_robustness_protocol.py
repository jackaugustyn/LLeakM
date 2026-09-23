from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_validation.scripts.robustness_eval import (
    AdaptiveLengthDecoder,
    common_decoder_seed,
    split_rows,
    transform_trace,
)
from weiss_reconstruction import reconstruct


class RobustnessProtocolTests(unittest.TestCase):
    def test_decoder_seed_is_common_across_conditions(self) -> None:
        seed = common_decoder_seed(20260706, 17)
        self.assertEqual(seed, common_decoder_seed(20260706, 17))
        self.assertNotEqual(seed, common_decoder_seed(20260707, 17))

    def test_split_is_prompt_disjoint_and_topic_balanced(self) -> None:
        rows = [
            {"idx": idx, "topic": topic, "response_complete": True}
            for topic in ("a", "b")
            for idx in range(1 if topic == "a" else 11, 5 if topic == "a" else 15)
        ]
        test, calibration = split_rows(rows, test_per_topic=2, complete_only=True)
        self.assertEqual({row["idx"] for row in test} & {row["idx"] for row in calibration}, set())
        self.assertEqual([row["topic"] for row in test].count("a"), 2)
        self.assertEqual([row["topic"] for row in test].count("b"), 2)

    def test_bucket_inverse_is_learned_only_from_calibration_pairs(self) -> None:
        decoder = AdaptiveLengthDecoder("bucket_8")
        decoder.fit([[1, 1, 7, 9, 9, 15]], seed=3)
        self.assertEqual(decoder.decode([8, 16]), [1, 9])

    def test_batch_inverse_preserves_observed_sums(self) -> None:
        decoder = AdaptiveLengthDecoder("batch_2")
        decoder.fit([[2, 3, 4, 5]], seed=3)
        decoded = decoder.decode([5, 9])
        self.assertEqual(decoded, [2, 3, 4, 5])
        self.assertEqual(transform_trace("batch_2", decoded, None), [5, 9])

    def test_reconstruct_device_env_cpu(self) -> None:
        from weiss_reconstruction import _choose_device, _MODEL_CACHE

        _MODEL_CACHE.clear()
        with patch.dict(os.environ, {"LLEAKM_RECONSTRUCT_DEVICE": "cpu"}, clear=False):
            self.assertEqual(_choose_device(), "cpu")
        lengths = ([2] * 9 + [1]) * 4
        with patch("weiss_reconstruction._load_model_bundle", return_value=(object(), object(), "cpu")), \
                patch("weiss_reconstruction._sample_and_rank", return_value=[("segment.", 0.0)]):
            full = reconstruct(lengths, max_sentences=0)
            limited = reconstruct(lengths, max_sentences=2)
        self.assertEqual(full.available_segment_count, 4)
        self.assertEqual(full.sentence_count, 4)
        self.assertEqual(limited.sentence_count, 2)


if __name__ == "__main__":
    unittest.main()
