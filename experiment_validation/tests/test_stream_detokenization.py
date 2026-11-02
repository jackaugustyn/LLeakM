"""Regression tests for contextual decoding and recorded stream provenance."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer, decoders, models
import app


class TestTokenizer:
    """Small real ByteFallback/Metaspace decoder, without model weights."""
    name_or_path = "test/contextual-byte-fallback"
    init_kwargs = {}
    eos_token_id = 99

    def __init__(self):
        self.backend_tokenizer = Tokenizer(models.WordLevel({
            "▁Hello": 0, "▁world": 1, "<0xC3>": 2, "<0xA9>": 3,
            "!": 4, "<unk>": 5,
        }, unk_token="<unk>"))
        self.backend_tokenizer.decoder = decoders.Sequence([
            decoders.Replace("▁", " "), decoders.ByteFallback(),
            decoders.Fuse(), decoders.Strip(" ", 1, 0),
        ])

    def decode(self, ids, **kwargs):
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)

    def __call__(self, text, return_tensors):
        return {"input_ids": torch.tensor([[0]], dtype=torch.long)}


class StreamDetokenizationTests(unittest.TestCase):
    def test_context_recovers_spaces_lost_by_individual_decode(self):
        tokenizer = TestTokenizer()
        self.assertEqual("".join(tokenizer.decode([x]) for x in [0, 1]), "Helloworld")
        fragments, reference = app._decode_token_fragments(tokenizer, [0, 1])
        self.assertEqual(reference, "Hello world")
        self.assertEqual(fragments, ["Hello", " world"])

    def test_partial_utf8_bytes_are_buffered_without_replacement_corruption(self):
        fragments, reference = app._decode_token_fragments(TestTokenizer(), [0, 1, 2, 3, 4])
        self.assertEqual(fragments, ["Hello", " world", "", "é", "!"])
        self.assertEqual(reference, "Hello worldé!")
        self.assertEqual(sum(len(x.encode()) for x in fragments), len(reference.encode()))

    def test_literal_replacement_at_end_is_retained(self):
        fragments, reference = app._decode_token_fragments(TestTokenizer(), [2])
        self.assertEqual(fragments, ["�"])
        self.assertEqual(reference, "�")

    def test_empty_response(self):
        self.assertEqual(app._decode_token_fragments(TestTokenizer(), []), ([], ""))

    def test_nonmonotone_decoder_is_rejected_before_streaming(self):
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "a" if len(ids) == 1 else "b")
        with self.assertRaisesRegex(ValueError, "not prefix-stable"):
            app._decode_token_fragments(tokenizer, [0, 1])

    def test_endpoint_logs_exact_ids_and_validated_reference_before_done(self):
        async def exercise(directory):
            with patch.object(app, "_tokenizer", TestTokenizer()), \
                 patch.object(app, "_model", SimpleNamespace(config=SimpleNamespace())), \
                 patch.object(app, "DEVICE", "cpu"), \
                 patch.object(app, "LOG_DIR", directory), \
                 patch.object(app, "_generate_one_by_one", return_value=([0, 1, 2, 3, 4], "eos")):
                response = await app.generate_sse("hello", 10, 0.0, 1.0, "legacy")
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)
                    if chunk == b"data: [DONE]\n\n":
                        self.assertTrue((Path(directory) / (response.headers["X-Run-Id"] + ".json")).is_file())
                log = json.loads((Path(directory) / (response.headers["X-Run-Id"] + ".json")).read_text())
                self.assertEqual(log["generated_token_ids"], [0, 1, 2, 3, 4])
                self.assertEqual([x["token_id"] for x in log["steps"]], [0, 1, 2, 3, 4])
                self.assertEqual(log["response_text"], "Hello worldé!")
                self.assertEqual(log["steps"][2]["token_utf8_len"], 0)
                self.assertEqual(b"".join(chunks), b"".join(app._sse_frame(x["token_text"]) for x in log["steps"]) + app._sse_frame("[DONE]"))
                self.assertEqual(log["harness_source_sha256"], app.HARNESS_SOURCE_SHA256)
                self.assertEqual(len(log["tokenizer_metadata"]["backend_serialization_sha256"]), 64)
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(exercise(directory))


if __name__ == "__main__":
    unittest.main()
