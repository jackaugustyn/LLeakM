from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from weiss_reconstruction import reconstruct_from_frame_lengths

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, replace))
        prev = curr
    return prev[-1]


def rouge1_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    ref = tokenize_words(reference)
    pred = tokenize_words(prediction)
    if not ref or not pred:
        return 0.0, 0.0, 0.0

    rc = Counter(ref)
    pc = Counter(pred)
    overlap = sum((rc & pc).values())

    recall = overlap / max(1, len(ref))
    precision = overlap / max(1, len(pred))
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return precision, recall, f1


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ai in a:
        curr = [0]
        for j, bj in enumerate(b, start=1):
            if ai == bj:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def rouge_l_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    ref = tokenize_words(reference)
    pred = tokenize_words(prediction)
    if not ref or not pred:
        return 0.0, 0.0, 0.0

    lcs = lcs_len(ref, pred)
    recall = lcs / len(ref)
    precision = lcs / len(pred)
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return precision, recall, f1


def first_sentence(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    m = re.search(r"(.+?[.!?])(?:\s|$)", t, flags=re.S)
    if m:
        return m.group(1).strip()
    return t.splitlines()[0].strip()


class SemanticScorer:
    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self.model = None

        if backend in {"auto", "minilm"}:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",
                    cache_folder=None,
                )
                self.backend = "minilm"
                return
            except Exception:
                if backend == "minilm":
                    raise

        self.backend = "tfidf"

    def cosine(self, a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0

        if self.backend == "minilm":
            va = self.model.encode([a], normalize_embeddings=True)
            vb = self.model.encode([b], normalize_embeddings=True)
            return float((va @ vb.T)[0][0])

        ta = tokenize_words(a)
        tb = tokenize_words(b)
        if not ta or not tb:
            return 0.0

        ca = Counter(ta)
        cb = Counter(tb)
        vocab = set(ca) | set(cb)
        dot = float(sum(ca[w] * cb[w] for w in vocab))
        na = math.sqrt(sum(v * v for v in ca.values()))
        nb = math.sqrt(sum(v * v for v in cb.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)


@dataclass
class PromptCase:
    idx: int
    topic: str
    prompt: str


@dataclass
class SampleResult:
    idx: int
    topic: str
    prompt: str
    victim_model_id: str
    run_id: str
    frame_count: int
    token_count: int
    response_text: str
    response_first_sentence: str
    pred_first_sentence: str
    pred_full_text: str
    phi_cosine: float
    ed_norm: float
    rouge1_precision: float
    rouge1_recall: float
    rouge1_f1: float
    rougeL_precision: float
    rougeL_recall: float
    rougeL_f1: float
    first_phi_cosine: float
    first_ed_norm: float
    first_rouge1_precision: float
    first_rougeL_precision: float


def load_prompts(prompts_dir: Path) -> list[PromptCase]:
    topic_queues: dict[str, deque[str]] = {}
    for path in sorted(prompts_dir.glob("*.txt")):
        topic = path.stem
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        topic_queues[topic] = deque(lines)

    # Round-robin interleave across topics to keep balanced ordering.
    ordered: list[PromptCase] = []
    idx = 1
    topics = sorted(topic_queues)
    while True:
        moved = False
        for topic in topics:
            q = topic_queues[topic]
            if not q:
                continue
            prompt = q.popleft()
            ordered.append(PromptCase(idx=idx, topic=topic, prompt=prompt))
            idx += 1
            moved = True
        if not moved:
            break

    return ordered


def stream_frame_lengths(base_url: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float, timeout: int) -> tuple[list[int], str]:
    params = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    frame_lens: list[int] = []
    with requests.get(base_url, params=params, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        run_id = resp.headers.get("X-Run-Id", "")
        for raw_line in resp.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            frame_lens.append(len(raw_line + b"\n\n"))
            if raw_line.startswith(b"data: [DONE]"):
                break
    return frame_lens, run_id


def wait_for_log(log_dir: Path, run_id: str, timeout_s: float = 10.0) -> dict:
    if not run_id:
        raise RuntimeError("Missing run_id in response header X-Run-Id")

    path = log_dir / f"{run_id}.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            with path.open(encoding="utf-8") as f:
                return json.load(f)
        time.sleep(0.05)
    raise FileNotFoundError(f"Log for run_id={run_id} not found within {timeout_s}s: {path}")


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(statistics.fmean(vals))


def pct(cond_count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * cond_count / total


def summarize(rows: list[SampleResult]) -> dict:
    total = len(rows)

    summary = {
        "count": total,
        "phi_mean": safe_mean(r.phi_cosine for r in rows),
        "ed_norm_mean": safe_mean(r.ed_norm for r in rows),
        "rouge1_precision_mean": safe_mean(r.rouge1_precision for r in rows),
        "rougeL_precision_mean": safe_mean(r.rougeL_precision for r in rows),
        "first_phi_mean": safe_mean(r.first_phi_cosine for r in rows),
        "first_ed_norm_mean": safe_mean(r.first_ed_norm for r in rows),
        "thresholds_percent": {
            "ED_eq_0": pct(sum(1 for r in rows if r.ed_norm == 0.0), total),
            "ED_le_0_1": pct(sum(1 for r in rows if r.ed_norm <= 0.1), total),
            "R1_eq_1": pct(sum(1 for r in rows if r.rouge1_precision >= 1.0), total),
            "R1_ge_0_9": pct(sum(1 for r in rows if r.rouge1_precision >= 0.9), total),
            "RL_eq_1": pct(sum(1 for r in rows if r.rougeL_precision >= 1.0), total),
            "RL_ge_0_9": pct(sum(1 for r in rows if r.rougeL_precision >= 0.9), total),
            "phi_eq_1": pct(sum(1 for r in rows if r.phi_cosine >= 0.999999), total),
            "phi_gt_0_9": pct(sum(1 for r in rows if r.phi_cosine > 0.9), total),
            "ASR_phi_gt_0_5": pct(sum(1 for r in rows if r.phi_cosine > 0.5), total),
        },
    }
    return summary


def by_topic(rows: list[SampleResult]) -> list[dict]:
    bucket: dict[str, list[SampleResult]] = defaultdict(list)
    for r in rows:
        bucket[r.topic].append(r)

    out: list[dict] = []
    for topic in sorted(bucket):
        grp = bucket[topic]
        n = len(grp)
        out.append(
            {
                "topic": topic,
                "count": n,
                "phi_mean": safe_mean(r.phi_cosine for r in grp),
                "ed_norm_mean": safe_mean(r.ed_norm for r in grp),
                "rouge1_precision_mean": safe_mean(r.rouge1_precision for r in grp),
                "rougeL_precision_mean": safe_mean(r.rougeL_precision for r in grp),
                "ASR_phi_gt_0_5": pct(sum(1 for r in grp if r.phi_cosine > 0.5), n),
            }
        )
    return out


def write_outputs(out_dir: Path, rows: list[SampleResult], summary_obj: dict, topic_obj: list[dict], cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    with (out_dir / "summary_by_topic.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topic",
                "count",
                "phi_mean",
                "ed_norm_mean",
                "rouge1_precision_mean",
                "rougeL_precision_mean",
                "ASR_phi_gt_0_5",
            ],
        )
        writer.writeheader()
        writer.writerows(topic_obj)

    md = []
    md.append("# Validation Report")
    md.append("")
    md.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    md.append("")
    md.append("## Global Summary")
    md.append("")
    md.append(f"- samples: {summary_obj['count']}")
    md.append(f"- phi_mean: {summary_obj['phi_mean']:.4f}")
    md.append(f"- ed_norm_mean: {summary_obj['ed_norm_mean']:.4f}")
    md.append(f"- rouge1_precision_mean: {summary_obj['rouge1_precision_mean']:.4f}")
    md.append(f"- rougeL_precision_mean: {summary_obj['rougeL_precision_mean']:.4f}")
    md.append(f"- ASR (phi > 0.5): {summary_obj['thresholds_percent']['ASR_phi_gt_0_5']:.2f}%")
    md.append("")
    md.append("## Threshold Metrics [%]")
    md.append("")
    for k, v in summary_obj["thresholds_percent"].items():
        md.append(f"- {k}: {v:.2f}")
    md.append("")
    md.append("## Per-Topic")
    md.append("")
    md.append("| topic | count | phi_mean | ed_norm_mean | R1_prec_mean | RL_prec_mean | ASR(phi>0.5) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in topic_obj:
        md.append(
            "| {topic} | {count} | {phi_mean:.4f} | {ed_norm_mean:.4f} | {rouge1_precision_mean:.4f} | {rougeL_precision_mean:.4f} | {ASR_phi_gt_0_5:.2f}% |".format(
                **row
            )
        )

    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Large-scale validation for Weiss-style reconstruction")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/generate_sse")
    parser.add_argument("--prompts-dir", default=str(Path(__file__).resolve().parents[1] / "prompts"))
    parser.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--max-prompts", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--samples-per-segment", type=int, default=3)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--num-first-candidates", type=int, default=3)
    parser.add_argument("--semantic-backend", choices=["auto", "minilm", "tfidf"], default="auto")
    parser.add_argument("--label", default="", help="Optional label appended to output run directory name")
    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir)
    results_root = Path(args.results_dir)
    log_dir = Path(args.log_dir)

    cases = load_prompts(prompts_dir)
    if args.max_prompts > 0:
        cases = cases[: args.max_prompts]

    scorer = SemanticScorer(backend=args.semantic_backend)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.label}" if args.label else ""
    out_dir = results_root / f"run_{ts}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[SampleResult] = []

    for i, case in enumerate(cases, start=1):
        frame_lens, run_id = stream_frame_lengths(
            args.base_url,
            prompt=case.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.timeout,
        )

        raw_log = wait_for_log(log_dir=log_dir, run_id=run_id)
        response_text = raw_log.get("response_text", "")
        victim_model_id = raw_log.get("model_id", "")

        result = reconstruct_from_frame_lengths(
            frame_lens,
            num_first_candidates=args.num_first_candidates,
            max_sentences=args.max_sentences,
            samples_per_segment=args.samples_per_segment,
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

        if i % 10 == 0 or i == len(cases):
            progress_path = out_dir / "progress.json"
            progress_payload = {
                "done": i,
                "total": len(cases),
                "last_topic": case.topic,
                "last_run_id": run_id,
            }
            progress_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{len(cases)}] topic={case.topic} run_id={run_id} phi={phi:.3f} ed={ed_norm:.3f}")

    summary_obj = summarize(rows)
    topic_obj = by_topic(rows)

    cfg = {
        "base_url": args.base_url,
        "prompts_dir": str(prompts_dir),
        "log_dir": str(log_dir),
        "max_prompts": len(cases),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "samples_per_segment": args.samples_per_segment,
        "max_sentences": args.max_sentences,
        "num_first_candidates": args.num_first_candidates,
        "semantic_backend_requested": args.semantic_backend,
        "semantic_backend_used": scorer.backend,
        "label": args.label,
        "victim_model_ids": sorted({r.victim_model_id for r in rows if r.victim_model_id}),
    }

    write_outputs(out_dir=out_dir, rows=rows, summary_obj=summary_obj, topic_obj=topic_obj, cfg=cfg)

    print(f"DONE: {len(rows)} samples")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
