"""A checkpoint's RNG use must not change a resumed record's sample."""
import random
import unittest
from unittest.mock import patch

import weiss_reconstruction as wr


class ReconstructionSeedResumeTests(unittest.TestCase):
    def test_cold_and_warm_checkpoint_caches_share_per_record_rng(self):
        loaded = set()

        def load(name):
            if name not in loaded:
                # Model initialization may consume randomness before weights load.
                for _ in range(23):
                    random.random()
                loaded.add(name)
            return object(), object(), 'cpu'

        def sample(*args, **kwargs):
            return [(f'draw {random.random():.17g}', 0.0)]

        with patch.object(wr, '_load_model_bundle', side_effect=load), \
             patch.object(wr, 'seed_reconstructor', side_effect=random.seed), \
             patch.object(wr, '_sample_and_rank', side_effect=sample):
            cold = wr.reconstruct([4]*12, max_sentences=1, seed=721)
            warm = wr.reconstruct([4]*12, max_sentences=1, seed=721)
            loaded.clear()
            resumed = wr.reconstruct([4]*12, max_sentences=1, seed=721)
        self.assertEqual(cold.full_text, warm.full_text)
        self.assertEqual(warm.full_text, resumed.full_text)


if __name__ == '__main__':
    unittest.main()
