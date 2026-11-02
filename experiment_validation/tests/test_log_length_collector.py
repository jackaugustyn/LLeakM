"""Observation invariants that the former line-based collector violated."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment_validation.scripts.run_validation import (
    DONE_WRITE, stream_frame_lengths, validated_log_frame_lengths,
)
from weiss_reconstruction import frame_lengths_to_token_lengths


FRAGMENTS = ["hello", ":\n\n", "\r\n", "", "żółć🙂", "[DONE]", "\n\ndata: [DONE]\n\n", " tail"]


def make_log(fragments=FRAGMENTS):
    text = "".join(fragments)
    return {
        "run_id": "test-run", "prompt": "test prompt", "generated_tokens": len(fragments),
        "response_text": text, "response_utf8_len": len(text.encode()),
        "steps": [dict(step=i, token_text=t, token_utf8_len=len(t.encode()),
                       sse_frame_utf8_len=len(t.encode()) + 8)
                  for i, t in enumerate(fragments)],
    }


class FakeResponse:
    def __init__(self, content, chunk_size):
        self.content, self.chunk_size = content, chunk_size
        self.headers = {"X-Run-Id": "test-run"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b""
        for start in range(0, len(self.content), self.chunk_size):
            yield self.content[start:start + self.chunk_size]

    def iter_lines(self, **kwargs):
        raise AssertionError("The collector must not infer event boundaries from lines")


class LogLengthCollectorTests(unittest.TestCase):
    def test_payload_bytes_survive_arbitrary_http_chunks(self):
        wire = b"".join(b"data: " + t.encode() + b"\n\n" for t in FRAGMENTS) + DONE_WRITE
        expected = [len(t.encode()) for t in FRAGMENTS]
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test-run.json").write_text(json.dumps(make_log()))
            for chunk_size in (1, 2, 7, 65536):
                with self.subTest(chunk_size=chunk_size), patch(
                    "experiment_validation.scripts.run_validation.requests.get",
                    return_value=FakeResponse(wire, chunk_size),
                ):
                    frames, run_id = stream_frame_lengths(
                        "http://example.invalid/generate_sse", "test prompt", 96,
                        0.0, 1.0, 10, log_dir=Path(td),
                    )
                    self.assertEqual(run_id, "test-run")
                    self.assertEqual(frame_lengths_to_token_lengths(frames), expected)
                    self.assertEqual(sum(frames), len(wire))

    def test_inconsistent_log_is_rejected(self):
        for field in ("token_utf8_len", "sse_frame_utf8_len", "step"):
            raw = make_log()
            raw["steps"][1][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                validated_log_frame_lengths(raw)
        for field, value in (("response_text", "other"), ("response_utf8_len", 1), ("generated_tokens", 1)):
            raw = make_log()
            raw[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validated_log_frame_lengths(raw)

    def test_mismatched_request_and_truncated_wire_are_rejected(self):
        raw = make_log()
        for kwargs in ({"run_id": "wrong"}, {"prompt": "wrong"}, {"streamed_bytes": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validated_log_frame_lengths(raw, **kwargs)

    def test_empty_generation_has_only_terminal_write(self):
        frames = validated_log_frame_lengths(make_log([]), streamed_bytes=len(DONE_WRITE))
        self.assertEqual(frames, [len(DONE_WRITE)])
        self.assertEqual(frame_lengths_to_token_lengths(frames), [])


if __name__ == "__main__":
    unittest.main()
