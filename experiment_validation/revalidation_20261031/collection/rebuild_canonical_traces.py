"""Rebuild exact historical emitted-text lengths; no generation or inference.

These validated vectors repair collection of the *stored emitted fragments*.
They do not restore missing word boundaries, recover original token IDs, or
attest which vector a historical reconstructor received.
"""
from pathlib import Path
import collections
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiment_validation.scripts.run_validation import validated_log_frame_lengths
from weiss_reconstruction import frame_lengths_to_token_lengths, heuristic_sentences_from_lengths

OUT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def legacy_replay(steps):
    wire = b"".join(b"data: " + s["token_text"].encode() + b"\n\n" for s in steps) + b"data: [DONE]\n\n"
    frames = []
    for line in wire.splitlines():
        if not line:
            continue
        frames.append(len(line) + 2)
        if line.startswith(b"data: [DONE]"):
            break
    return frame_lengths_to_token_lengths(frames)


def main():
    models = json.loads((ROOT / "experiment_validation/analysis/model_stats.json").read_text())
    summaries, all_records, errors = [], [], []
    common_prompts = None
    source_hashes = {}
    for model in models:
        sample_file = ROOT / "experiment_validation/results" / model["run_dir"] / "samples.jsonl"
        sample_sha = digest(sample_file.read_bytes())
        source_hashes[str(sample_file.relative_to(ROOT))] = sample_sha
        records = [json.loads(line) for line in sample_file.read_text().splitlines() if line]
        prompts = [(r["idx"], r["topic"], r["prompt"]) for r in records]
        if common_prompts is None:
            common_prompts = prompts
        assert prompts == common_prompts, model["label"]
        assert len(records) == model["n"] == 300
        counts = collections.Counter()
        for sample in records:
            log_file = ROOT / "logs" / (sample["run_id"] + ".json")
            try:
                raw_bytes = log_file.read_bytes()
                log = json.loads(raw_bytes)
                frames = validated_log_frame_lengths(log, run_id=sample["run_id"], prompt=sample["prompt"])
                assert sample["response_text"] == log["response_text"], "sample/log reference mismatch"
                assert sample["victim_model_id"] == log["model_id"], "model mismatch"
                assert sample["token_count"] == log["generated_tokens"], "token-count mismatch"
                assert sample["finish_reason"] == log["finish_reason"], "finish-reason mismatch"
                assert sample["response_complete"] == log["response_complete"], "EOS mismatch"
                exact = frame_lengths_to_token_lengths(frames)
                replay = legacy_replay(log["steps"])
                counts["validated_samples"] += 1
                counts["changed_traces_vs_current_legacy_replay"] += exact != replay
                counts["changed_segment_counts_vs_legacy_replay"] += len(heuristic_sentences_from_lengths(exact)) != len(heuristic_sentences_from_lengths(replay))
                counts["references_without_ascii_spaces"] += " " not in log["response_text"]
                counts["references_with_unicode_replacement_character"] += "\ufffd" in log["response_text"]
                counts["total_emitted_fragments"] += len(exact)
                counts["total_payload_bytes"] += sum(exact)
                counts["empty_fragments"] += exact.count(0)
                all_records.append({
                    "label": model["label"], "model": model["model"],
                    "idx": sample["idx"], "topic": sample["topic"], "prompt": sample["prompt"],
                    "run_id": sample["run_id"], "run_dir": model["run_dir"],
                    "sample_file_sha256": sample_sha,
                    "generation_log_path": str(log_file.relative_to(ROOT)),
                    "generation_log_sha256": digest(raw_bytes),
                    "token_lengths": exact,
                    "token_lengths_sha256": digest(json.dumps(exact, separators=(",", ":")).encode()),
                    "reference": log["response_text"],
                    "reference_sha256": digest(log["response_text"].encode()),
                    "old_pred_full_text": sample["pred_full_text"],
                    "reconstruction_seed": sample["reconstruction_seed"],
                    "finish_reason": log["finish_reason"], "response_complete": log["response_complete"],
                    "legacy_replay_differs": exact != replay,
                    "reference_validation": "validated_emitted_text_only; original-token detokenization unverified",
                    "diagnostic_series": model["label"] in {"tinyllama_1_1b", "phi_3_5_mini"},
                    "observation_source": "validated-historical-log-payload-lengths-v1",
                })
            except (ValueError, AssertionError, OSError) as exc:
                errors.append({"label": model["label"], "idx": sample["idx"], "run_id": sample["run_id"], "error": str(exc)})
        summaries.append({"label": model["label"], "run_dir": model["run_dir"], **dict(counts)})
    assert not errors, errors
    assert len(all_records) == 2100
    output = OUT / "canonical_traces.jsonl"
    pending_output = output.with_suffix(".jsonl.tmp")
    with pending_output.open("w") as handle:
        for row in all_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    pending_output.replace(output)
    payload = {
        "scope": "Exact byte lengths of archived emitted fragments; no new inference, no reference repair, no historical attacker-input attestation.",
        "all_log_consistency_checks_passed": True,
        "sample_count": len(all_records), "models": summaries,
        "source_sample_hashes": source_hashes,
        "canonical_trace_file_sha256": digest(output.read_bytes()),
        "validation_source_sha256": digest((ROOT / "experiment_validation/scripts/run_validation.py").read_bytes()),
        "legacy_replay_source": "previous iter_lines split/filter/add-newline logic, with terminal-prefix stop",
        "errors": errors,
    }
    (OUT / "canonical_trace_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"sample_count": len(all_records), "models": summaries}, indent=2))


if __name__ == "__main__":
    main()
