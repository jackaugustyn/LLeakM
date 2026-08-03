"""Trace-information baselines: shuffled / constant / cross-prompt lengths.

Quantifies how much of reconstruction quality is due to the real length trace
versus the reconstructor's language prior. Runs on the same topic-balanced
subset used for defenses (default: 2 prompts/topic x 2 models).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiment_validation" / "results"
LOGDIR = ROOT / "logs"
OUT = ROOT / "experiment_validation" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-z0-9']+")

RUNS = [
    ("qwen_1_5b", "Qwen2.5-1.5B-Instruct", "run_20260325_104518_qwen_1_5b"),
    ("llama_3_2_3b", "Llama-3.2-3B-Instruct", "run_20260414_073030_llama_3_2_3b"),
]


def tf_cosine(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    ta, tb = WORD_RE.findall(a.lower()), WORD_RE.findall(b.lower())
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    vocab = set(ca) | set(cb)
    dot = float(sum(ca[w] * cb[w] for w in vocab))
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def balanced_subset(rows: list[dict], per_topic: int) -> list[dict]:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    chosen = []
    for topic in sorted(by_topic):
        grp = sorted(by_topic[topic], key=lambda r: r["idx"])
        chosen.extend(grp[:per_topic])
    chosen.sort(key=lambda r: r["idx"])
    return chosen


def load_samples(run_dir: str) -> list[dict]:
    rows = []
    with (RESULTS / run_dir / "samples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            rows.append({"idx": o["idx"], "topic": o["topic"],
                         "run_id": o["run_id"], "response_text": o.get("response_text", "")})
    return rows


def token_lengths_for(run_id: str) -> list[int] | None:
    p = LOGDIR / f"{run_id}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return [int(s["token_utf8_len"]) for s in d.get("steps", [])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=2)
    ap.add_argument("--samples-per-segment", type=int, default=3)
    ap.add_argument("--max-sentences", type=int, default=3)
    ap.add_argument("--num-first-candidates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from weiss_reconstruction import reconstruct

    out_path = OUT / "trace_baseline_samples.jsonl"
    # resume support
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                done.add((o["label"], int(o["idx"]), o["condition"]))
    mode = "a" if done else "w"
    out_f = out_path.open(mode, encoding="utf-8")
    if done:
        print(f"[baseline] resuming; {len(done)} triples done", flush=True)

    t0 = time.time()
    n_done = 0
    # Collect all length sequences per model for cross-prompt baseline
    all_toks: dict[str, list[list[int]]] = {}

    for label, pretty, run_dir in RUNS:
        rows = load_samples(run_dir)
        subset = balanced_subset(rows, args.per_topic)
        if args.limit:
            subset = subset[: args.limit]
        toks_list = []
        for r in subset:
            t = token_lengths_for(r["run_id"])
            if t:
                toks_list.append(t)
        all_toks[label] = toks_list
        print(f"[baseline] {pretty}: {len(subset)} prompts", flush=True)

        for r in subset:
            toks = token_lengths_for(r["run_id"])
            if not toks:
                continue
            gt = r["response_text"]
            conditions = {}
            # real
            conditions["real"] = list(toks)
            # shuffled within-response
            rng = np.random.default_rng(args.seed + r["idx"] * 17)
            sh = list(toks)
            rng.shuffle(sh)
            conditions["shuffled"] = sh
            # constant = mean length
            m = max(1, int(round(float(np.mean(toks)))))
            conditions["constant"] = [m] * len(toks)
            # cross-prompt: lengths from another prompt in the subset (cyclic)
            others = [t for t in all_toks[label] if t is not toks]
            if others:
                donor = others[r["idx"] % len(others)]
                # resample to same length by cycling
                conditions["cross_prompt"] = [donor[i % len(donor)] for i in range(len(toks))]
            else:
                conditions["cross_prompt"] = list(toks)

            for cname, seq in conditions.items():
                if (label, r["idx"], cname) in done:
                    continue
                seed = (args.seed + r["idx"] * 131 + hash(cname) % 9973) & 0x7fffffff
                torch.manual_seed(seed)
                np.random.seed(seed % (2**32 - 1))
                rec = reconstruct(
                    seq,
                    num_first_candidates=args.num_first_candidates,
                    max_sentences=args.max_sentences,
                    samples_per_segment=args.samples_per_segment,
                )
                pred = rec.full_text
                phi = tf_cosine(gt, pred)
                out_f.write(json.dumps({
                    "model": pretty, "label": label, "idx": r["idx"], "topic": r["topic"],
                    "condition": cname, "phi": phi, "pred": pred[:500],
                }, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
            print(f"[baseline] idx={r['idx']} done_new={n_done} elapsed={time.time()-t0:.0f}s", flush=True)

    out_f.close()

    # Aggregate
    by = defaultdict(list)
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by[(o["label"], o["condition"])].append(float(o["phi"]))

    summary = []
    rng = np.random.default_rng(args.seed)
    for label, pretty, _ in RUNS:
        real = np.array(by.get((label, "real"), []))
        for cname in ["real", "shuffled", "constant", "cross_prompt"]:
            arr = np.array(by.get((label, cname), []))
            if len(arr) == 0:
                continue
            n = len(arr)
            idx = rng.integers(0, n, size=(10000, n))
            lo, hi = float(np.percentile(arr[idx].mean(axis=1), [2.5, 97.5]))
            delta = float(arr.mean() - real.mean()) if len(real) else float("nan")
            summary.append({
                "model": pretty, "label": label, "condition": cname, "n": n,
                "phi_mean": float(arr.mean()), "phi_ci": [lo, hi],
                "delta_vs_real": delta,
            })
    (OUT / "trace_baselines.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["% Trace-information baselines"]
    prev = None
    for s in summary:
        if s["model"] != prev and prev is not None:
            lines.append("\\midrule")
        cell = s["model"] if s["model"] != prev else ""
        prev = s["model"]
        dlt = s["delta_vs_real"]
        dlt_s = "---" if s["condition"] == "real" else f"{dlt:+.4f}"
        lines.append(
            f"{cell} & {s['condition'].replace('_', '\\_')} & {s['n']} & "
            f"{s['phi_mean']:.4f} & {{[{s['phi_ci'][0]:.4f}, {s['phi_ci'][1]:.4f}]}} & "
            f"{dlt_s} \\\\")
    (OUT / "tables_trace_baselines.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[baseline] DONE new={n_done} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
