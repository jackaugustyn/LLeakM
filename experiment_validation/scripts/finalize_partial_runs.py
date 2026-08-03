from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from weiss_reconstruction import reconstruct_from_frame_lengths

from experiment_validation.scripts.run_validation import (
    PromptCase,
    SampleResult,
    SemanticScorer,
    by_topic,
    first_sentence,
    levenshtein_distance,
    load_prompts,
    rouge1_scores,
    rouge_l_scores,
    stream_frame_lengths,
    summarize,
    wait_for_log,
    write_outputs,
)


DONE_FRAME_LEN = len(b"data: [DONE]\n\n")
DEFAULT_RESULTS_ROOT = ROOT_DIR / "experiment_validation" / "results"
DEFAULT_PROMPTS = ROOT_DIR / "experiment_validation" / "prompts"
DEFAULT_PROMPTS_REMAINING_150 = ROOT_DIR / "experiment_validation" / "prompts_remaining_150"
DEFAULT_LOG_DIR = ROOT_DIR / "logs"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _iter_partial_runs(results_root: Path) -> list[Path]:
    out: list[Path] = []
    for d in sorted(results_root.glob("run_*")):
        if not d.is_dir():
            continue
        if not (d / "progress.json").exists():
            continue
        if (d / "summary.json").exists():
            continue
        out.append(d)
    return out


def _load_prompt_cases(run_dir: Path, total: int) -> list[PromptCase]:
    if total == 150 or "remaining150" in run_dir.name:
        base = DEFAULT_PROMPTS_REMAINING_150
    else:
        base = DEFAULT_PROMPTS

    cases = load_prompts(base)
    if len(cases) < total:
        raise ValueError(f"{run_dir.name}: prompt set has only {len(cases)} prompts, expected at least {total}")
    return cases[:total]


def _valid_qwen3b_log(path: Path) -> bool:
    try:
        data = _read_json(path)
    except Exception:
        return False
    return (
        data.get("model_id") == MODEL_ID
        and data.get("max_new_tokens") == 96
        and float(data.get("temperature", 0.0)) == 0.0
        and float(data.get("top_p", 1.0)) == 1.0
    )


def _build_log_index(log_dir: Path) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(log_dir.glob("*.json")):
        if not _valid_qwen3b_log(path):
            continue
        data = _read_json(path)
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            continue
        index[prompt].append(data)
    return index


def _log_to_frame_lens(log_obj: dict) -> list[int]:
    steps = log_obj.get("steps", [])
    frame_lens = [int(s.get("sse_frame_utf8_len", 0)) for s in steps if int(s.get("sse_frame_utf8_len", 0)) > 0]
    frame_lens.append(DONE_FRAME_LEN)
    return frame_lens


def _collect_missing_prompts(cases_by_run: dict[Path, list[PromptCase]], log_index: dict[str, list[dict]]) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for cases in cases_by_run.values():
        for c in cases:
            p = c.prompt.strip()
            if p in log_index:
                continue
            if p in seen:
                continue
            seen.add(p)
            missing.append(p)
    return missing


def _ensure_server(base_url: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["MODEL_ID"] = MODEL_ID
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"]
    srv = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env)
    time.sleep(8)
    return srv


def _generate_missing_logs(
    base_url: str,
    missing_prompts: list[str],
    log_dir: Path,
    timeout: int,
) -> dict[str, dict]:
    generated: dict[str, dict] = {}
    if not missing_prompts:
        return generated

    srv = _ensure_server(base_url=base_url)
    try:
        for i, prompt in enumerate(missing_prompts, start=1):
            _, run_id = stream_frame_lengths(
                base_url=base_url,
                prompt=prompt,
                max_new_tokens=96,
                temperature=0.0,
                top_p=1.0,
                timeout=timeout,
            )
            raw_log = wait_for_log(log_dir=log_dir, run_id=run_id, timeout_s=30.0)
            generated[prompt] = raw_log
            if i % 10 == 0 or i == len(missing_prompts):
                print(f"[missing {i}/{len(missing_prompts)}] run_id={run_id}")
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=20)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait(timeout=10)

    return generated


def _build_rows_for_run(
    run_dir: Path,
    cases: list[PromptCase],
    log_index: dict[str, list[dict]],
    semantic_backend: str,
) -> tuple[list[SampleResult], dict]:
    scorer = SemanticScorer(backend=semantic_backend)
    rows: list[SampleResult] = []
    last_run_id = ""

    for i, case in enumerate(cases, start=1):
        entries = log_index.get(case.prompt.strip())
        if not entries:
            raise RuntimeError(f"{run_dir.name}: missing log for prompt idx={case.idx}")
        raw_log = entries[-1]

        frame_lens = _log_to_frame_lens(raw_log)
        run_id = raw_log.get("run_id", "")
        response_text = raw_log.get("response_text", "")
        victim_model_id = raw_log.get("model_id", "")

        result = reconstruct_from_frame_lengths(
            frame_lens,
            num_first_candidates=3,
            max_sentences=3,
            samples_per_segment=3,
        )

        pred_text = result.full_text
        gt_first = first_sentence(response_text)
        pred_first = result.first_sentence

        ed = levenshtein_distance(response_text, pred_text)
        ed_norm = ed / max(1, len(response_text))

        r1_p, r1_r, r1_f1 = rouge1_scores(response_text, pred_text)
        rl_p, rl_r, rl_f1 = rouge_l_scores(response_text, pred_text)
        phi = scorer.cosine(response_text, pred_text)

        fed = levenshtein_distance(gt_first, pred_first)
        fed_norm = fed / max(1, len(gt_first))
        fr1_p, _, _ = rouge1_scores(gt_first, pred_first)
        frl_p, _, _ = rouge_l_scores(gt_first, pred_first)
        fphi = scorer.cosine(gt_first, pred_first)

        row = SampleResult(
            idx=case.idx,
            topic=case.topic,
            prompt=case.prompt,
            victim_model_id=victim_model_id,
            run_id=run_id,
            frame_count=len(frame_lens),
            token_count=max(0, len(frame_lens) - 1),
            response_text=response_text,
            response_first_sentence=gt_first,
            pred_first_sentence=pred_first,
            pred_full_text=pred_text,
            phi_cosine=phi,
            ed_norm=ed_norm,
            rouge1_precision=r1_p,
            rouge1_recall=r1_r,
            rouge1_f1=r1_f1,
            rougeL_precision=rl_p,
            rougeL_recall=rl_r,
            rougeL_f1=rl_f1,
            first_phi_cosine=fphi,
            first_ed_norm=fed_norm,
            first_rouge1_precision=fr1_p,
            first_rougeL_precision=frl_p,
        )
        rows.append(row)
        last_run_id = run_id

        if i % 25 == 0 or i == len(cases):
            print(f"[{run_dir.name}] {i}/{len(cases)}")

    cfg = {
        "base_url": "http://127.0.0.1:8000/generate_sse",
        "prompts_dir": str(DEFAULT_PROMPTS_REMAINING_150 if (len(cases) == 150 or "remaining150" in run_dir.name) else DEFAULT_PROMPTS),
        "log_dir": str(DEFAULT_LOG_DIR),
        "max_prompts": len(cases),
        "max_new_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "samples_per_segment": 3,
        "max_sentences": 3,
        "num_first_candidates": 3,
        "semantic_backend_requested": semantic_backend,
        "semantic_backend_used": scorer.backend,
        "label": "qwen_3b",
        "victim_model_ids": [MODEL_ID],
        "finalized_from_partial": run_dir.name,
    }
    if last_run_id:
        (run_dir / "progress.json").write_text(
            json.dumps({"done": len(cases), "total": len(cases), "last_topic": rows[-1].topic, "last_run_id": last_run_id}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return rows, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize partial qwen_3b runs into complete report artifacts")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/generate_sse")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--semantic-backend", choices=["auto", "minilm", "tfidf"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    log_dir = Path(args.log_dir)
    partial_runs = _iter_partial_runs(results_root=results_root)

    if not partial_runs:
        print("No partial runs found.")
        return

    cases_by_run: dict[Path, list[PromptCase]] = {}
    for run_dir in partial_runs:
        p = _read_json(run_dir / "progress.json")
        total = int(p.get("total", 0))
        if total <= 0:
            raise ValueError(f"{run_dir.name}: invalid total in progress.json")
        cases_by_run[run_dir] = _load_prompt_cases(run_dir=run_dir, total=total)

    log_index = _build_log_index(log_dir=log_dir)
    missing_prompts = _collect_missing_prompts(cases_by_run=cases_by_run, log_index=log_index)

    print(f"Partial runs: {len(partial_runs)}")
    print(f"Missing prompts to generate: {len(missing_prompts)}")

    if args.dry_run:
        return

    generated = _generate_missing_logs(
        base_url=args.base_url,
        missing_prompts=missing_prompts,
        log_dir=log_dir,
        timeout=args.timeout,
    )
    for prompt, raw in generated.items():
        log_index[prompt].append(raw)

    for run_dir in partial_runs:
        rows, cfg = _build_rows_for_run(
            run_dir=run_dir,
            cases=cases_by_run[run_dir],
            log_index=log_index,
            semantic_backend=args.semantic_backend,
        )
        summary_obj = summarize(rows)
        topic_obj = by_topic(rows)
        write_outputs(out_dir=run_dir, rows=rows, summary_obj=summary_obj, topic_obj=topic_obj, cfg=cfg)
        print(f"Finalized: {run_dir}")


if __name__ == "__main__":
    main()
