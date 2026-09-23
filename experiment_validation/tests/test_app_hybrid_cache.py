from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import _uses_hybrid_cache


class HybridCacheRoutingTests(unittest.TestCase):
    def test_gemma2_uses_hybrid_cache(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma2"))
        self.assertTrue(_uses_hybrid_cache(model))

    def test_explicit_hybrid_implementation(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="llama", cache_implementation="hybrid")
        )
        self.assertTrue(_uses_hybrid_cache(model))

    def test_llama_keeps_growing_cache(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
        self.assertFalse(_uses_hybrid_cache(model))


if __name__ == "__main__":
    unittest.main()
