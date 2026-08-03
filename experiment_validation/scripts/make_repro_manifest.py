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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment_validation" / "results"
ANALYSIS = ROOT / "experiment_validation" / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

STANDARDIZED_RUNS = [
    "run_20260325_104518_qwen_1_5b",
    "run_20260327_124817_qwen_3b_final_resume",
    "run_20260414_073030_llama_3_2_3b",
    "run_20260415_080034_gemma_2_2b_it",
    "run_20260329_085714_phi_1_5",
    "run_20260325_153010_tinyllama_1_1b",
    "run_20260414_111644_phi_3_5_mini",
]

RUN_FILES = ["config.json", "progress.json", "samples.jsonl",
             "summary.json", "summary.md", "summary_by_topic.csv"]

CODE_FILES = [
    "app.py", "client.py", "weiss_reconstruction.py",
    "experiment_validation/scripts/run_validation.py",
    "experiment_validation/scripts/run_model_matrix.py",
    "experiment_validation/scripts/generate_prompts.py",
    "experiment_validation/scripts/analyze_stats.py",
    "experiment_validation/scripts/defense_eval.py",
    "experiment_validation/scripts/make_repro_manifest.py",
    "experiment_validation/scripts/make_figures.py",
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
]

PACKAGES = ["numpy", "torch", "transformers", "sentence-transformers",
            "scipy", "scikit-learn", "requests", "fastapi", "uvicorn"]


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

    analysis_hashes = {}
    for rel in ANALYSIS_FILES:
        p = ROOT / "experiment_validation" / rel
        if p.exists():
            analysis_hashes[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": pkg_versions(),
        },
        "victim_models": [
            "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-2b-it",
            "microsoft/phi-1_5", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "microsoft/Phi-3.5-mini-instruct",
        ],
        "reconstruction_models": [
            "royweiss1/T5_FirstSentences", "royweiss1/T5_MiddleSentences",
        ],
        "seed": 20260706,
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
            "stats": "python experiment_validation/scripts/analyze_stats.py",
            "defenses": ("HF_HUB_OFFLINE=1 python experiment_validation/scripts/defense_eval.py "
                         "--per-topic 2 --samples-per-segment 3 --max-sentences 3"),
            "figures": "python experiment_validation/scripts/make_figures.py",
            "manifest": "python experiment_validation/scripts/make_repro_manifest.py",
        },
    }

    (ANALYSIS / "repro_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md = ["# Reproducibility Package", "",
          f"Generated (UTC): {manifest['generated_utc']}",
          f"Git commit: `{manifest['git_commit']}`", "",
          "## Environment", "",
          f"- Python: {manifest['environment']['python']}",
          f"- Platform: {manifest['environment']['platform']}"]
    for k, v in manifest["environment"]["packages"].items():
        md.append(f"- {k}: {v}")
    md += ["", "## Standardized protocol", ""]
    for k, v in manifest["standardized_protocol"].items():
        md.append(f"- {k}: {v}")
    md += ["", "## Regeneration commands", ""]
    for k, v in manifest["commands"].items():
        md.append(f"- **{k}**: `{v}`")
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
