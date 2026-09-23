"""Build a reproducibility manifest for the study.

Captures everything a reviewer needs to reproduce and verify the results:

  * SHA-256 hashes of every run artifact in the standardized subset and of the
    analysis outputs, so the exact bytes behind every reported number can be
    checked.
  * A frozen software environment (Python, key package versions, platform).
  * The exact command lines used to regenerate prompts, traces, statistics, and
    defenses.

Writes ``experiment_validation/analysis/repro_manifest.json`` and a
human-readable ``experiment_validation/analysis/REPRODUCIBILITY.md``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment_validation" / "results"
ANALYSIS = ROOT / "experiment_validation" / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

STANDARDIZED_RUNS = [
    "run_full_20260920_qwen_1_5b_full",
    "run_full_20260920_qwen_3b_full",
    "run_full_20260920_llama_3_2_3b_full",
    "run_full_20260920_gemma_2_2b_full",
    "run_full_20260920_phi_1_5_full",
    "run_full_20260920_tinyllama_1_1b_full",
    "run_full_20260920_phi_3_5_mini_full",
]

RUN_FILES = ["config.json", "progress.json", "samples.jsonl",
             "summary.json", "summary.md", "summary_by_topic.csv"]

CODE_FILES = [
    "app.py", "client.py", "weiss_reconstruction.py",
    "experiment_validation/scripts/run_validation.py",
    "experiment_validation/scripts/run_model_matrix.py",
    "experiment_validation/scripts/generate_prompts.py",
    "experiment_validation/scripts/analyze_stats.py",
    "experiment_validation/scripts/analyze_extras.py",
    "experiment_validation/scripts/defense_eval.py",
    "experiment_validation/scripts/robustness_eval.py",
    "experiment_validation/scripts/trace_baselines.py",
    "experiment_validation/scripts/make_repro_manifest.py",
    "experiment_validation/scripts/make_figures.py",
    "experiment_validation/scripts/audit_release_content.py",
    "experiment_validation/scripts/verify_manuscript_numbers.py",
    "experiment_validation/models_publication.json",
    "experiment_validation/tests/test_robustness_protocol.py",
]

ANALYSIS_FILES = [
    "analysis/model_stats.json", "analysis/pairwise_tests.json",
    "analysis/per_prompt_phi.csv", "analysis/tables.tex",
    "analysis/tables_main.tex", "analysis/tables_pairwise.tex",
    "analysis/tables_topic.tex",
    "analysis/defense_eval.json", "analysis/defense_tables.tex",
    "analysis/defense_tables_body.tex", "analysis/defense_pairwise_tests.json",
    "analysis/defense_pairwise.tex", "analysis/defense_samples.jsonl",
    "analysis/fig_phi_ci.tex", "analysis/fig_defense.tex",
    "analysis/release_content_audit.json", "analysis/manuscript_number_verification.json",
    "analysis/env_snapshot.json",
    "analysis/robustness_eval.json", "analysis/robustness_samples.jsonl",
    "analysis/robustness_samples_config.json", "analysis/robustness_table.tex",
]

PACKAGES = ["numpy", "torch", "transformers", "sentence-transformers",
            "scipy", "scikit-learn", "requests", "fastapi", "uvicorn",
            "protobuf", "sentencepiece"]

VICTIM_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b-it",
    "microsoft/phi-1_5", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3.5-mini-instruct",
]
RECONSTRUCTION_MODELS = [
    "royweiss1/T5_FirstSentences", "royweiss1/T5_MiddleSentences",
]
TOKENIZER_CLASS_FALLBACK = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2Tokenizer",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2Tokenizer",
    "meta-llama/Llama-3.2-3B-Instruct": "PreTrainedTokenizerFast",
    "google/gemma-2-2b-it": "GemmaTokenizer",
    "microsoft/phi-1_5": "CodeGenTokenizer",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "LlamaTokenizer",
    "microsoft/Phi-3.5-mini-instruct": "LlamaTokenizer",
    "royweiss1/T5_FirstSentences": "T5Tokenizer",
    "royweiss1/T5_MiddleSentences": "T5Tokenizer",
}

COLLECTION_ENVIRONMENT = {
    "python": "3.12.3",
    "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
    "machine": "x86_64",
    "packages": {
        "numpy": "2.4.3",
        "torch": "2.11.0+cu130",
        "transformers": "5.3.0",
        "sentence-transformers": "not installed",
        "scipy": "not installed",
        "scikit-learn": "not installed",
        "requests": "2.32.5",
        "fastapi": "0.135.2",
        "uvicorn": "0.42.0",
    },
    "note": "Frozen environment used to collect traces and compute the publication statistics.",
}

SEED = 20260706
N_BOOT = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_versions() -> dict[str, str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return {}
    out = {}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except Exception:
            out[p] = "not installed"
    return out


def hardware_info() -> dict:
    info: dict = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "uname": platform.uname()._asdict(),
    }
    try:
        mem = Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in mem.splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                info["mem_total_gb"] = round(kb / (1024 ** 2), 2)
                break
        cpu = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in cpu.splitlines():
            if line.lower().startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True).strip()
        gpus = []
        for row in gpu.splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "name": parts[0], "driver": parts[1],
                    "memory": parts[2], "compute_cap": parts[3],
                })
        info["gpus"] = gpus
    except Exception:
        info["gpus"] = []
    try:
        import torch
        info["torch_cuda_available"] = bool(torch.cuda.is_available())
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["torch_cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch_cuda_available"] = False
    return info


def _hub_dir(model_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / (
        "models--" + model_id.replace("/", "--")
    )


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def model_revisions(model_ids: list[str]) -> dict[str, dict]:
    """Capture local Hugging Face snapshot hashes and tokenizer metadata."""
    out: dict[str, dict] = {}
    for mid in model_ids:
        rec: dict = {
            "id": mid, "local_ref": None, "local_snapshots": [],
            "tokenizer_class": None, "tokenizer_revision": None, "source": None,
        }
        hub = _hub_dir(mid)
        ref = hub / "refs" / "main"
        if ref.exists():
            rec["local_ref"] = ref.read_text(encoding="utf-8").strip()
            rec["source"] = "huggingface_hub_cache"
        snap_root = hub / "snapshots"
        if snap_root.exists():
            rec["local_snapshots"] = sorted(
                p.name for p in snap_root.iterdir() if p.is_dir()
            )
            snap = rec["local_ref"] or (
                rec["local_snapshots"][0] if rec["local_snapshots"] else None
            )
            if snap:
                tok = _read_json(snap_root / snap / "tokenizer_config.json")
                rec["tokenizer_class"] = tok.get("tokenizer_class")
                rec["tokenizer_revision"] = snap
                cfg = _read_json(snap_root / snap / "config.json")
                rec["model_type"] = cfg.get("model_type")
                rec["architectures"] = cfg.get("architectures")
        if rec["tokenizer_class"] is None:
            rec["tokenizer_class"] = TOKENIZER_CLASS_FALLBACK.get(mid)
        if rec["local_ref"] is None:
            rec["source"] = rec["source"] or "not_in_local_cache"
            rec["hub_sha_at_audit"] = _hub_sha(mid)
            rec["hub_sha_note"] = (
                "Hub main SHA at audit time; the original collection snapshot "
                "is no longer in the local cache."
            )
        out[mid] = rec
    return out


def _hub_sha(model_id: str) -> str | None:
    """Best-effort current Hub commit; not necessarily the collection revision."""
    try:
        import urllib.request
        url = f"https://huggingface.co/api/models/{model_id}"
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data.get("sha") or data.get("xetSha")
    except Exception:
        return None


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    artifacts = {}
    for run in STANDARDIZED_RUNS:
        run_hashes = {}
        for fn in RUN_FILES:
            p = RESULTS / run / fn
            if p.exists():
                run_hashes[fn] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        artifacts[run] = run_hashes

    code_hashes = {}
    for rel in CODE_FILES:
        p = ROOT / rel
        if p.exists():
            code_hashes[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    hw = hardware_info()
    victim_revs = model_revisions(VICTIM_MODELS)
    recon_revs = model_revisions(RECONSTRUCTION_MODELS)
    analysis_packages = pkg_versions()
    try:
        import numpy as np
        analysis_packages["numpy"] = np.__version__
    except Exception:
        pass
    analysis_env = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": analysis_packages,
        "hardware": hw,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "note": "Host used to re-run publication analyses from released artifacts.",
    }
    env = {
        "collection": COLLECTION_ENVIRONMENT,
        "analysis_rerun": analysis_env,
        "python": COLLECTION_ENVIRONMENT["python"],
        "platform": COLLECTION_ENVIRONMENT["platform"],
        "machine": COLLECTION_ENVIRONMENT["machine"],
        "packages": COLLECTION_ENVIRONMENT["packages"],
        "hardware": hw,
        "pythonhashseed": analysis_env["pythonhashseed"],
    }
    (ANALYSIS / "env_snapshot.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    analysis_hashes = {}
    for rel in ANALYSIS_FILES:
        p = ROOT / "experiment_validation" / rel
        if p.exists():
            analysis_hashes[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": env,
        "victim_models": VICTIM_MODELS,
        "reconstruction_models": RECONSTRUCTION_MODELS,
        "model_revisions": victim_revs,
        "reconstruction_revisions": recon_revs,
        "seeds": {
            "analysis": SEED,
            "bootstrap_resamples": N_BOOT,
            "victim_decoding_temperature": 0.0,
            "victim_decoding_top_p": 1.0,
            "robustness_decoder_seed_indices": [20260706, 20260707, 20260708, 20260709, 20260710],
            "robustness_common_random_numbers": True,
            "robustness_trace_seed": 20260920,
            "note": (
                "Publication statistics use numpy.random.Generator(SEED). "
                "Defense reconstruction used per-sample seeds "
                "(SEED + idx * 131 + hash(defense) % 9973). "
                "Set PYTHONHASHSEED=0 before replaying randomized padding."
            ),
        },
        "seed": SEED,
        "standardized_protocol": {
            "N": 300, "max_new_tokens": 96, "samples_per_segment": 3,
            "max_sentences": 3, "num_first_candidates": 3,
            "temperature": 0.0, "top_p": 1.0,
            "semantic_backend": "term-frequency cosine (fixed, recomputed for all runs)",
        },
        "artifacts": artifacts,
        "code": code_hashes,
        "analysis_outputs": analysis_hashes,
        "commands": {
            "prompts": "python experiment_validation/scripts/generate_prompts.py",
            "collect": ("HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_validation.py "
                        "--max-prompts 300 --max-new-tokens 96 --samples-per-segment 3 --max-sentences 3"),
            "stats": "PYTHONHASHSEED=0 python experiment_validation/scripts/analyze_stats.py",
            "extras": "PYTHONHASHSEED=0 python experiment_validation/scripts/analyze_extras.py",
            "defenses": ("HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 python experiment_validation/scripts/defense_eval.py "
                         "--per-topic 2 --samples-per-segment 3 --max-sentences 3"),
            "robustness_plan": "python experiment_validation/scripts/robustness_eval.py --plan",
            "repeated_adaptive_evaluation": (
                "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                "python experiment_validation/scripts/robustness_eval.py"
            ),
            "complete_response_collection": (
                "python experiment_validation/scripts/run_model_matrix.py "
                "--models-file experiment_validation/models_publication.json --prompts 300 "
                "--max-new-tokens 512 --samples-per-segment 3 --max-sentences 0 "
                "--num-first-candidates 3 --prompt-format legacy "
                "--reconstruction-seed 20260706 --hf-offline"
            ),
            "figures": "python experiment_validation/scripts/make_figures.py",
            "audit": "python experiment_validation/scripts/audit_release_content.py",
            "verify": "python experiment_validation/scripts/verify_manuscript_numbers.py",
            "manifest": "python experiment_validation/scripts/make_repro_manifest.py",
        },
    }

    (ANALYSIS / "repro_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# Reproducibility Package", "",
          f"Generated (UTC): {manifest['generated_utc']}",
          f"Git commit: `{manifest['git_commit']}`", "",
          "## Environment", "",
          f"- Collection Python: {COLLECTION_ENVIRONMENT['python']}",
          f"- Collection platform: {COLLECTION_ENVIRONMENT['platform']}",
          f"- Analysis re-run Python: {analysis_env['python']}",
          f"- Analysis re-run platform: {analysis_env['platform']}",
          f"- PYTHONHASHSEED: `{manifest['environment']['pythonhashseed']}`"]
    hw = manifest["environment"]["hardware"]
    if hw.get("cpu_model"):
        md.append(f"- CPU: {hw['cpu_model']} ({hw.get('cpu_count')} threads)")
    if hw.get("mem_total_gb"):
        md.append(f"- Memory: {hw['mem_total_gb']} GiB")
    for gpu in hw.get("gpus") or []:
        md.append(f"- GPU: {gpu['name']} (driver {gpu['driver']}, {gpu['memory']}, compute {gpu['compute_cap']})")
    md.append("- Collection packages:")
    for k, v in COLLECTION_ENVIRONMENT["packages"].items():
        md.append(f"  - {k}: {v}")
    md.append("- Analysis re-run packages:")
    for k, v in analysis_env["packages"].items():
        md.append(f"  - {k}: {v}")
    md += ["", "## Seeds", "",
           f"- Analysis / bootstrap seed: `{SEED}`",
           f"- Bootstrap resamples: `{N_BOOT}`",
           f"- Victim decoding: temperature `{0.0}`, top_p `{1.0}`",
           "",
           "The robustness extension uses decoder seed indices `20260706` through `20260710`. "
           "For each prompt and seed index, `blake2s(\"decoder\", seed_index, prompt_idx)` "
           "produces one condition-independent PyTorch/NumPy/Python seed. Randomized-padding "
           "noise has a separate trace seed and remains fixed across decoder repetitions.",
           "", "## Model and tokenizer revisions", "",
           "Local Hugging Face snapshot hashes are recorded when the checkpoint is still in the cache. "
           "`not_in_local_cache` means the collection machine no longer holds that snapshot; "
           "the Hub identifier remains the publication pin.",
           "",
           "| Model | Snapshot | Tokenizer class | Source |",
           "|---|---|---|---|"]
    for mid, rec in {**victim_revs, **recon_revs}.items():
        snap = rec.get("local_ref") or rec.get("tokenizer_revision") or rec.get("hub_sha_at_audit") or "n/a"
        md.append(
            f"| `{mid}` | `{snap}` | "
            f"{rec.get('tokenizer_class') or 'n/a'} | {rec.get('source')} |"
        )
    md += ["", "## Standardized protocol", ""]
    for k, v in manifest["standardized_protocol"].items():
        md.append(f"- {k}: {v}")
    md += ["", "## Regeneration commands", ""]
    for k, v in manifest["commands"].items():
        md.append(f"- **{k}**: `{v}`")
    md += ["", "## Robustness-extension safeguards", "",
           "- The default test split contains the first two aligned prompts per topic; all remaining prompts are calibration-only.",
           "- The adaptive MAP inverse is fitted per victim model and defense without using test responses.",
           "- `max_sentences=0` decodes every available segment.",
           "- Complete-response claims require `response_complete=true`, emitted only for EOS before the cap.",
           "- The robustness LaTeX table is emitted only after every expected prompt-by-seed cell is present."]
    md += ["", "## Artifact integrity (SHA-256)", "",
           "Each standardized run directory contains six artifacts; hashes are in "
           "`repro_manifest.json` under `artifacts`. Analysis outputs and code are "
           "hashed under `analysis_outputs` and `code`.", ""]
    md.append("| Run | samples.jsonl SHA-256 (prefix) | bytes |")
    md.append("|---|---|---:|")
    for run, files in manifest["artifacts"].items():
        s = files.get("samples.jsonl", {})
        md.append(f"| {run} | `{s.get('sha256','')[:16]}...` | {s.get('bytes','')} |")
    (ANALYSIS / "REPRODUCIBILITY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[repro] wrote manifest ({len(artifacts)} runs, {len(code_hashes)} code files)")


if __name__ == "__main__":
    main()
