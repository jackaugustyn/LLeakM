"""Provenance and byte-accounting checks; these never load victim weights."""
from __future__ import annotations
import asyncio
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("generate_fresh", HERE / "generate_fresh.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from experiment_validation.tests.test_stream_detokenization import TestTokenizer
import app


class FreshReferenceTests(unittest.TestCase):
    def run_fixture(self, directory):
        async def generate():
            with patch.object(app, "_tokenizer", TestTokenizer()), \
                 patch.object(app, "_model", SimpleNamespace(config=SimpleNamespace())), \
                 patch.object(app, "DEVICE", "cpu"), \
                 patch.object(app, "LOG_DIR", str(directory)), \
                 patch.object(app, "_generate_one_by_one", return_value=([0, 1, 2, 3, 4], "eos")):
                response = await app.generate_sse("hello", 512, 0.0, 1.0, "legacy")
                chunks = [chunk async for chunk in response.body_iterator]
                path = directory / (response.headers["X-Run-Id"] + ".json")
                return json.loads(path.read_text()), chunks, path
        return asyncio.run(generate())

    def test_ids_and_contextual_bytes_are_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, chunks, _ = self.run_fixture(Path(temporary))
            self.assertEqual(runner.validate_log(raw, TestTokenizer(), "hello", chunks), [5, 6, 0, 2, 1])
            self.assertEqual(raw["response_text"], "Hello worldé!")

    def test_corrupt_fragments_ids_or_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw, chunks, _ = self.run_fixture(Path(temporary))
            for mutation in ["id", "bytes", "reference", "frame"]:
                changed = copy.deepcopy(raw)
                changed_chunks = list(chunks)
                if mutation == "id": changed["generated_token_ids"][0] = 4
                if mutation == "bytes": changed["steps"][0]["token_utf8_len"] = 99
                if mutation == "reference": changed["response_text"] = "corrupted"
                if mutation == "frame": changed_chunks[0] = b"truncated"
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    runner.validate_log(changed, TestTokenizer(), "hello", changed_chunks)

    def test_resume_is_bound_to_source_and_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw, chunks, log_path = self.run_fixture(directory)
            lengths = runner.validate_log(raw, TestTokenizer(), "hello", chunks)
            source = dict(label="test", idx=1, topic="topic", prompt="hello", reconstruction_seed=123)
            record = dict(source, protocol_id="p", run_id=raw["run_id"], reference=raw["response_text"],
                          generated_token_ids=raw["generated_token_ids"], token_lengths=lengths,
                          reference_sha256=runner.sha_bytes(raw["response_text"].encode()),
                          token_lengths_sha256=runner.sha_bytes(json.dumps(lengths, separators=(",", ":")).encode()),
                          generation_log_sha256=runner.sha_file(log_path))
            runner.validate_resume_record(record, source, "p", directory)
            changed = dict(record, prompt="changed")
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                runner.validate_resume_record(changed, source, "p", directory)
            log_path.write_text(log_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "log hash"):
                runner.validate_resume_record(record, source, "p", directory)

    def test_complete_cli_load_contract_limit_resume_and_corruption(self):
        """Exercise the real handler and output path with mocked weights/CUDA."""
        class Tokenizer(TestTokenizer):
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[0]]) if return_tensors else [0]}

        class Model:
            config = SimpleNamespace()
            hf_device_map = {"model.layers.0": 0, "model.layers.1": "cpu"}

            def eval(self): return self

        original_handler = app.generate_sse

        async def cpu_handler(*args):
            with patch.object(app, "DEVICE", "cpu"):
                return await original_handler(*args)

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            directory = Path(temporary)
            dummy_weight = directory / "mock_checkpoint"
            dummy_weight.write_text("This is a test fixture, not a model checkpoint")
            for name, value in [("is_available", True), ("get_device_name", "mock CUDA"), ("memory_allocated", 0)]:
                stack.enter_context(patch("torch.cuda." + name, return_value=value))
            for name in ["set_num_threads", "set_num_interop_threads", "manual_seed"]:
                stack.enter_context(patch("torch." + name))
            stack.enter_context(patch("torch.cuda.manual_seed_all"))
            stack.enter_context(patch("huggingface_hub.hf_hub_download", return_value=str(dummy_weight)))
            stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=Tokenizer()))
            loader = stack.enter_context(patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=Model()))
            stack.enter_context(patch.object(app, "generate_sse", side_effect=cpu_handler))
            stack.enter_context(patch.object(app, "_generate_one_by_one", return_value=([0, 1, 2, 3, 4], "eos")))
            # Keep injected module globals local to this test.
            for name in ["MODEL_ID", "DEVICE", "DTYPE", "LOG_DIR", "_tokenizer", "_model"]:
                stack.enter_context(patch.object(app, name, getattr(app, name)))
            output = directory / "fresh"
            argv = ["generate_fresh.py", "--label", "phi_3_5_mini", "--output", str(output),
                    "--placement", "auto_offload", "--gpu-memory-gib", "6", "--limit", "1"]
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(runner.main(), 0)
            self.assertEqual(loader.call_args.kwargs["device_map"], "auto")
            self.assertEqual(loader.call_args.kwargs["max_memory"][0], 6 * 1024**3)
            self.assertEqual(loader.call_args.kwargs["torch_dtype"], torch.float16)
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertEqual(json.loads((output / "status.json").read_text())["completed"], 1)
            self.assertEqual(runner.main(), 0)
            records = [json.loads(line) for line in (output / "canonical_traces.jsonl").read_text().splitlines()]
            self.assertEqual([record["idx"] for record in records], [1, 2])
            self.assertEqual([record["reference"] for record in records], ["Hello worldé!"] * 2)
            self.assertTrue(all(not record["historical_comparison_available"] for record in records))
            self.assertTrue(all(record["source_kind"] == "fresh_contextual_reference" for record in records))
            records[0]["reference"] = "corrupted"
            (output / "canonical_traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
            with self.assertRaisesRegex(ValueError, "reference hash"):
                runner.main()


if __name__ == "__main__":
    unittest.main()
