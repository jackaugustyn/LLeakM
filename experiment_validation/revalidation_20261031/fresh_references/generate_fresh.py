"""Generate fresh victim references with exact IDs and verified contextual decoding.

This is an instrumented generation-only rerun, not a network measurement. The
repaired app handler is drained in-process; no T5 model is loaded. Archived
prompts/settings are reused, but historical victim checkpoint identity is not
known. New reference/trace pairs must not be substituted silently into old rows.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def validate_log(raw, tokenizer, expected_prompt: str, chunks: list[bytes]) -> list[int]:
    """Verify IDs, full decode, token step accounting and exact emitted frames."""
    ids = raw["generated_token_ids"]
    steps = raw["steps"]
    if raw["prompt"] != expected_prompt:
        raise ValueError("Prompt mismatch")
    if raw["detokenization_policy"] != "contextual_prefix_utf8_buffer_v1":
        raise ValueError("Unrecognized detokenization policy")
    if len(ids) != len(steps) or raw["generated_tokens"] != len(ids):
        raise ValueError("Token count mismatch")
    if ids != [step["token_id"] for step in steps]:
        raise ValueError("Token ID mismatch")
    reference = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if reference != raw["response_text"] or reference != "".join(x["token_text"] for x in steps):
        raise ValueError("Full-sequence decoding mismatch")
    lengths = []
    expected_frames = []
    for index, step in enumerate(steps):
        length = len(step["token_text"].encode("utf-8"))
        if step["step"] != index or step["token_utf8_len"] != length:
            raise ValueError("Step index or payload-length mismatch")
        frame = ("data: " + step["token_text"] + "\n\n").encode("utf-8")
        if step["sse_frame_utf8_len"] != len(frame):
            raise ValueError("Frame-length mismatch")
        lengths.append(length)
        expected_frames.append(frame)
    expected_frames.append(b"data: [DONE]\n\n")
    if chunks != expected_frames:
        raise ValueError("Observed emitted frames disagree with verified log")
    if sum(lengths) != raw["response_utf8_len"] or sum(lengths) != len(reference.encode("utf-8")):
        raise ValueError("Response byte-count mismatch")
    return lengths


def validate_resume_record(record, source, protocol_id, logs_dir):
    if record["protocol_id"] != protocol_id:
        raise ValueError("Resumed protocol mismatch")
    for field in ["label", "idx", "topic", "prompt", "reconstruction_seed"]:
        if record[field] != source[field]:
            raise ValueError(f"Resumed source mismatch: {field}")
    if record["reference_sha256"] != sha_bytes(record["reference"].encode("utf-8")):
        raise ValueError("Resumed reference hash mismatch")
    if record["token_lengths_sha256"] != sha_bytes(json.dumps(record["token_lengths"], separators=(",", ":")).encode()):
        raise ValueError("Resumed length-vector hash mismatch")
    log_path = logs_dir / (record["run_id"] + ".json")
    if sha_file(log_path) != record["generation_log_sha256"]:
        raise ValueError("Resumed generation-log hash mismatch")
    raw = json.loads(log_path.read_text())
    if raw["response_text"] != record["reference"] or raw["generated_token_ids"] != record["generated_token_ids"]:
        raise ValueError("Resumed record and generation log disagree")
    if [step["token_utf8_len"] for step in raw["steps"]] != record["token_lengths"]:
        raise ValueError("Resumed record and generation lengths disagree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=["tinyllama_1_1b", "phi_3_5_mini"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=HERE.parent / "collection/canonical_traces.jsonl")
    parser.add_argument("--models", type=Path, default=HERE / "models_pinned.json")
    parser.add_argument("--placement", choices=["full_cuda", "auto_offload"], default="full_cuda")
    parser.add_argument("--gpu-memory-gib", type=float, default=6.0)
    parser.add_argument("--cpu-memory-gib", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="New records this invocation; zero means all pending.")
    parser.add_argument("--plan", action="store_true", help="Validate source and print plan without loading weights.")
    args = parser.parse_args()
    if args.limit < 0 or args.gpu_memory_gib <= 0 or args.cpu_memory_gib <= 0:
        raise ValueError("Invalid execution limits")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["label"] == args.label]
    if len(rows) != 300 or sorted(row["idx"] for row in rows) != list(range(1, 301)):
        raise ValueError("Fresh rerun requires exactly the archived 300 unique prompt indices")
    pins = json.loads(args.models.read_text())
    model_spec = next(item for item in pins["models"] if item["label"] == args.label)
    archived_config_path = ROOT / "experiment_validation/results" / rows[0]["run_dir"] / "config.json"
    archived_config = json.loads(archived_config_path.read_text())
    for key, value in {"max_new_tokens": 512, "temperature": 0.0, "top_p": 1.0, "prompt_format": "legacy"}.items():
        if archived_config[key] != value:
            raise ValueError(f"Archived generation contract differs: {key}")
    plan = {
        "expected": 300, "label": args.label, "model_id": model_spec["model_id"],
        "revision": model_spec["revision"], "generation": {key: archived_config[key] for key in ["max_new_tokens", "temperature", "top_p", "prompt_format"]},
        "precision": "float16", "placement_requested": args.placement,
        "max_memory_gib": {"cuda:0": args.gpu_memory_gib, "cpu": args.cpu_memory_gib} if args.placement == "auto_offload" else None,
        "canonical_input_sha256": sha_file(args.input), "archived_config_sha256": sha_file(archived_config_path),
        "models_file_sha256": sha_file(args.models), "runner_sha256": sha_file(Path(__file__)),
        "harness_source_sha256": sha_file(ROOT / "app.py"),
        "observation_source": "validated-in-process-harness-payload-byte-lengths-v1",
        "scope": "Fresh victim generations with verified ordinary full-sequence decoding; not historical IDs or checkpoint recovery; no network timing claim.",
        "historical_revision_identity": pins["historical_revision_identity"],
        "historical_prediction_comparison": "not applicable; canonical old_pred_full_text is an empty compatibility placeholder",
    }
    if args.plan:
        print(json.dumps(plan, indent=2)); return 0
    out = args.output.resolve()
    for protected in [ROOT / "experiment_validation/results", ROOT / "experiment_validation/analysis", ROOT / "logs"]:
        if out == protected or protected in out.parents:
            raise ValueError("Output must be separate from archived results, analysis and logs")
    import torch
    import transformers
    import accelerate
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit device or precision fallback")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    checkpoint_files = {
        name: sha_file(Path(hf_hub_download(model_spec["model_id"], name, revision=model_spec["revision"], local_files_only=True)))
        for name in model_spec["files"]
    }
    protocol = dict(plan, torch=torch.__version__, transformers=transformers.__version__, accelerate=accelerate.__version__,
                    checkpoint_file_sha256=checkpoint_files, cuda_device=torch.cuda.get_device_name(0), threads=4)
    protocol_id = sha_bytes(json.dumps(protocol, sort_keys=True).encode())
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)
    protocol_file = out / "protocol.json"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        raise ValueError("Existing output protocol differs; choose a separate output directory")
    write_json(protocol_file, protocol)
    os.environ["MODEL_ID"] = model_spec["model_id"]
    os.environ["LOG_DIR"] = str(log_dir)
    import app
    if app.HARNESS_SOURCE_SHA256 != protocol["harness_source_sha256"]:
        raise ValueError("Imported harness source differs from protocol")
    samples = out / "canonical_traces.jsonl"
    by_idx = {row["idx"]: row for row in rows}
    completed = {}
    if samples.exists():
        for line in samples.read_text().splitlines():
            record = json.loads(line)
            idx = record["idx"]
            if idx not in by_idx or idx in completed:
                raise ValueError("Unknown or duplicate resumed index")
            validate_resume_record(record, by_idx[idx], protocol_id, log_dir)
            completed[idx] = record
    status = dict(protocol_id=protocol_id, expected=300, completed=len(completed), complete=len(completed) == 300,
                  status="starting", new_records_this_invocation=0, scope=plan["scope"])
    write_json(out / "status.json", status)
    started = time.monotonic()
    try:
        if len(completed) < 300:
            status["status"] = "loading_pinned_model"
            write_json(out / "status.json", status)
            app.MODEL_ID = model_spec["model_id"]
            app.DEVICE = "cuda"
            app.DTYPE = torch.float16
            app.LOG_DIR = str(log_dir)
            app._tokenizer = AutoTokenizer.from_pretrained(model_spec["model_id"], revision=model_spec["revision"], local_files_only=True, use_fast=True)
            loader = dict(revision=model_spec["revision"], local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            if args.placement == "auto_offload":
                loader.update(device_map="auto", max_memory={0: int(args.gpu_memory_gib * 1024**3), "cpu": int(args.cpu_memory_gib * 1024**3)})
            else:
                loader["device_map"] = {"": "cuda:0"}
            app._model = AutoModelForCausalLM.from_pretrained(model_spec["model_id"], **loader).eval()
            device_map = {str(key): str(value) for key, value in app._model.hf_device_map.items()}
            if "disk" in device_map.values():
                raise ValueError("Disk offload not supported by this experiment protocol")
            runtime = dict(protocol_id=protocol_id, device_map=device_map,
                           tokenizer_metadata=app._tokenizer_metadata(), cuda_memory_allocated=torch.cuda.memory_allocated())
            runtime_file = out / "runtime.json"
            if runtime_file.exists() and json.loads(runtime_file.read_text())["device_map"] != device_map:
                raise ValueError("Actual device map changed on resume; separate output required")
            write_json(runtime_file, runtime)
            print("MODEL_READY " + json.dumps(runtime), flush=True)
        for row in rows:
            if row["idx"] in completed:
                continue
            if args.limit and status["new_records_this_invocation"] >= args.limit:
                break
            status.update(status="running", active_index=row["idx"])
            write_json(out / "status.json", status)
            torch.manual_seed(row["reconstruction_seed"])
            torch.cuda.manual_seed_all(row["reconstruction_seed"])
            sample_start = time.monotonic()
            async def generate():
                response = await app.generate_sse(row["prompt"], 512, 0.0, 1.0, "legacy")
                frames = [chunk async for chunk in response.body_iterator]
                return response.headers["X-Run-Id"], frames
            run_id, frames = asyncio.run(generate())
            log_path = log_dir / (run_id + ".json")
            raw = json.loads(log_path.read_text())
            lengths = validate_log(raw, app._tokenizer, row["prompt"], frames)
            if raw["harness_source_sha256"] != protocol["harness_source_sha256"]:
                raise ValueError("Generation log source hash mismatch")
            reference = raw["response_text"]
            input_text = app._build_input_text(row["prompt"], "legacy")
            record = dict(
                label=row["label"], model=row["model"], model_id=model_spec["model_id"], idx=row["idx"], topic=row["topic"], prompt=row["prompt"],
                run_id=run_id, run_dir=str(out), protocol_id=protocol_id, generation_log_path=str(log_path.relative_to(ROOT)) if ROOT in log_path.parents else str(log_path), generation_log_sha256=sha_file(log_path),
                token_lengths=lengths, token_lengths_sha256=sha_bytes(json.dumps(lengths, separators=(",", ":")).encode()),
                reference=reference, reference_sha256=sha_bytes(reference.encode("utf-8")), generated_token_ids=raw["generated_token_ids"],
                input_text=input_text, input_token_ids=app._tokenizer(input_text)["input_ids"],
                reconstruction_seed=row["reconstruction_seed"], old_pred_full_text="", historical_comparison_available=False,
                source_kind="fresh_contextual_reference", historical_run_id=row["run_id"], historical_reference_sha256=row["reference_sha256"],
                historical_sample_file_sha256=row["sample_file_sha256"], historical_run_dir=row["run_dir"],
                finish_reason=raw["finish_reason"], response_complete=raw["response_complete"],
                diagnostic_series=False, reference_validation="contextual-full-sequence-decoding-verified-v1; fresh pinned victim generation",
                observation_source=plan["observation_source"], emitted_stream_sha256=sha_bytes(b"".join(frames)),
                emitted_stream_bytes=sum(map(len, frames)), elapsed_seconds=time.monotonic() - sample_start,
            )
            with samples.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            completed[row["idx"]] = record
            status["new_records_this_invocation"] += 1
            status.update(completed=len(completed), complete=len(completed) == 300, elapsed_seconds=time.monotonic() - started)
            write_json(out / "status.json", status)
            print(f"{len(completed)}/300 {args.label} idx={row['idx']} tokens={len(lengths)} finish={record['finish_reason']} seconds={record['elapsed_seconds']:.1f}", flush=True)
        status["status"] = "completed" if status["complete"] else "partial"
        status.pop("active_index", None)
        write_json(out / "status.json", status)
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.monotonic() - started)
        write_json(out / "status.json", status)
        raise
    write_json(out / "summary.json", dict(
        complete=len(completed) == 300, expected=300, completed=len(completed), protocol_id=protocol_id,
        finish_reasons=dict(Counter(row["finish_reason"] for row in completed.values())),
        references_with_ascii_space=sum(" " in row["reference"] for row in completed.values()),
        empty_payload_steps=sum(value == 0 for row in completed.values() for value in row["token_lengths"]),
        scope=plan["scope"], historical_comparison_available=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
