from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "experiment_validation" / "results"
DEFAULT_MODELS_FILE = ROOT_DIR / "experiment_validation" / "models.json"


def run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> None:
    proc = subprocess.run(cmd, env=env, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation across multiple victim models")
    parser.add_argument("--models-file", default=str(DEFAULT_MODELS_FILE))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--prompts", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--samples-per-segment", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--num-first-candidates", type=int, default=3)
    parser.add_argument("--clear-results", action="store_true")
    parser.add_argument("--hf-offline", action="store_true")
    args = parser.parse_args()

    models = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
    if not isinstance(models, list) or not models:
        raise ValueError("models-file must be a non-empty JSON list")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_results:
        for item in results_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    for m in models:
        label = m["label"]
        model_id = m["model_id"]

        env = os.environ.copy()
        env["MODEL_ID"] = model_id
        if args.hf_offline:
            env["HF_HUB_OFFLINE"] = "1"

        uvicorn_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]

        print(f"\n=== MODEL: {label} ({model_id}) ===")
        srv = subprocess.Popen(uvicorn_cmd, env=env, cwd=str(ROOT_DIR))
        try:
            time.sleep(8)

            validation_cmd = [
                sys.executable,
                "experiment_validation/scripts/run_validation.py",
                "--max-prompts",
                str(args.prompts),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--samples-per-segment",
                str(args.samples_per_segment),
                "--max-sentences",
                str(args.max_sentences),
                "--num-first-candidates",
                str(args.num_first_candidates),
                "--label",
                label,
            ]
            run_cmd(validation_cmd, env=env, cwd=ROOT_DIR)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=20)
            except subprocess.TimeoutExpired:
                srv.kill()
                srv.wait(timeout=10)

    print("\nAll model runs completed.")


if __name__ == "__main__":
    main()
