from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT_DIR / "experiment_validation" / "results"
DEFAULT_MODELS_FILE = ROOT_DIR / "experiment_validation" / "models.json"


def run_cmd(cmd: list[str], env: dict[str, str], cwd: Path) -> None:
    proc = subprocess.run(cmd, env=env, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def wait_for_server(url: str, process: subprocess.Popen, timeout_s: int = 600) -> None:
    """Wait for model loading instead of assuming a fixed startup time."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"model server exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"model server did not become ready within {timeout_s}s: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation across multiple victim models")
    parser.add_argument("--models-file", default=str(DEFAULT_MODELS_FILE))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--prompts", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--samples-per-segment", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--num-first-candidates", type=int, default=3)
    parser.add_argument("--prompt-format", choices=["legacy", "chat_template"], default="legacy")
    parser.add_argument("--reconstruction-seed", type=int, default=20260706)
    parser.add_argument(
        "--campaign-id", default="",
        help="stable identifier used in run directories; reuse it with --resume",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--server-startup-timeout", type=int, default=600)
    parser.add_argument("--clear-results", action="store_true")
    parser.add_argument("--hf-offline", action="store_true")
    parser.add_argument(
        "--reconstruct-device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="device for the T5 reconstructor; cpu keeps an 8 GB victim GPU from OOM",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="local uvicorn port; default 8010 avoids a occupied :8000",
    )
    args = parser.parse_args()

    models = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
    if not isinstance(models, list) or not models:
        raise ValueError("models-file must be a non-empty JSON list")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    campaign_id = args.campaign_id or datetime.now().strftime("%Y%m%d_%H%M%S_full")
    print(f"Campaign ID: {campaign_id}", flush=True)

    if args.clear_results:
        for item in results_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    for m in models:
        label = m["label"]
        model_id = m["model_id"]
        output_dir = results_dir / f"run_{campaign_id}_{label}"
        progress_path = output_dir / "progress.json"
        if args.resume and progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if int(progress.get("done", 0)) >= args.prompts:
                print(f"\n=== SKIP COMPLETE: {label} ({output_dir}) ===", flush=True)
                continue

        env = os.environ.copy()
        env["MODEL_ID"] = model_id
        if args.hf_offline:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        if args.reconstruct_device != "auto":
            env["LLEAKM_RECONSTRUCT_DEVICE"] = args.reconstruct_device

        uvicorn_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]

        print(f"\n=== MODEL: {label} ({model_id}) ===")
        srv = subprocess.Popen(
            uvicorn_cmd, env=env, cwd=str(ROOT_DIR), start_new_session=True
        )
        try:
            wait_for_server(f"http://127.0.0.1:{args.port}/docs", srv, args.server_startup_timeout)

            validation_cmd = [
                sys.executable,
                "experiment_validation/scripts/run_validation.py",
                "--base-url",
                f"http://127.0.0.1:{args.port}/generate_sse",
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
                "--prompt-format",
                args.prompt_format,
                "--reconstruction-seed",
                str(args.reconstruction_seed),
                "--output-dir",
                str(output_dir),
                "--label",
                label,
            ]
            if args.resume:
                validation_cmd.append("--resume")
            run_cmd(validation_cmd, env=env, cwd=ROOT_DIR)
        finally:
            if srv.poll() is None:
                try:
                    os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    srv.terminate()
                try:
                    srv.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(srv.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        srv.kill()
                    srv.wait(timeout=5)

    print("\nAll model runs completed.")


if __name__ == "__main__":
    main()
