"""Defense-mechanism evaluation for the SSE frame-length side channel.

Each defense is a transformation applied to the per-token UTF-8 length sequence
that the attacker observes through SSE frame sizes. We reconstruct text from the
defended sequence using the same Weiss-style pipeline and measure the residual
leakage, so the drop relative to the undefended baseline quantifies each
mitigation's effectiveness.

Defenses implemented (all server-side, transport-preserving):
  * ``baseline``   -- observed token lengths, no mitigation (not a defense).
  * ``bucket_8``   -- quantize each token length up to a multiple of 8 bytes.
  * ``pad_32``     -- online fixed frame size C=32 bytes (tokens longer than C
                      still emit size C; no look-ahead over the full response).
  * ``batch_2`` / ``batch_4`` -- emit one frame per k tokens.
  * ``rand_pad_8`` -- add uniform random padding in [0, 8] bytes per token.

Five defense configurations (four mechanism classes: bucketing, fixed framing,
batching, randomized padding). Evaluated against a *fixed, non-adaptive*
reconstructor trained on undefended traces.

Ground truth is the victim ``response_text`` stored with each sample; token
lengths come from the per-run generation logs. Metrics use the same fixed
TF-IDF cosine backend as ``analyze_stats.py`` for cross-condition comparability.

The reconstruction step is stochastic (nucleus sampling in the T5 decoder), so
we seed torch/NumPy per sample for reproducibility.
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

# Victim runs used for the defense study (strong + moderate leakers).
DEFENSE_RUNS = [
    ("qwen_1_5b", "Qwen2.5-1.5B-Instruct", "run_20260325_104518_qwen_1_5b"),
    ("llama_3_2_3b", "Llama-3.2-3B-Instruct", "run_20260414_073030_llama_3_2_3b"),
]


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def tf_cosine(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    ta, tb = tokenize_words(a), tokenize_words(b)
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


# ------------------------- defenses -------------------------

def d_baseline(t: list[int], rng: np.random.Generator) -> list[int]:
    return list(t)


def d_bucket8(t: list[int], rng: np.random.Generator) -> list[int]:
    return [int(math.ceil(x / 8.0) * 8) for x in t]


def d_pad32(t: list[int], rng: np.random.Generator) -> list[int]:
    """Online fixed frame width C=32 (no response-wide look-ahead)."""
    return [32 for _ in t]


def _batch(t: list[int], k: int) -> list[int]:
    return [int(sum(t[i:i + k])) for i in range(0, len(t), k)]


def d_batch2(t: list[int], rng: np.random.Generator) -> list[int]:
    return _batch(t, 2)


def d_batch4(t: list[int], rng: np.random.Generator) -> list[int]:
    return _batch(t, 4)


def d_rand_pad8(t: list[int], rng: np.random.Generator) -> list[int]:
    return [int(x + rng.integers(0, 9)) for x in t]


DEFENSES = {
    "baseline": d_baseline,
    "bucket_8": d_bucket8,
    "pad_32": d_pad32,
    "batch_2": d_batch2,
    "batch_4": d_batch4,
    "rand_pad_8": d_rand_pad8,
}


def balanced_subset(rows: list[dict], per_topic: int, seed: int) -> list[dict]:
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    rng = np.random.default_rng(seed)
    chosen: list[dict] = []
    for topic in sorted(by_topic):
        grp = sorted(by_topic[topic], key=lambda r: r["idx"])
        take = grp[:per_topic]
        chosen.extend(take)
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


def levenshtein_norm(ref: str, pred: str) -> float:
    if not ref:
        return 0.0
    a, b = ref, pred
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1] / max(1, len(ref))


def rouge1_precision(ref: str, pred: str) -> float:
    r, p = tokenize_words(ref), tokenize_words(pred)
    if not r or not p:
        return 0.0
    overlap = sum((Counter(r) & Counter(p)).values())
    return overlap / max(1, len(p))


def load_completed(path: Path) -> set[tuple[str, int, str]]:
    """Return (label, idx, defense) triples already present on disk."""
    done: set[tuple[str, int, str]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            done.add((o["label"], int(o["idx"]), o["defense"]))
    return done


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> dict:
    """Two-sided paired Wilcoxon signed-rank test (normal approximation)."""
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "p_value": 1.0, "rank_biserial": 0.0}
    order = np.argsort(np.abs(d), kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sa = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    W = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        rb = (w_pos - w_neg) / (w_pos + w_neg)
        return {"n": n, "p_value": 1.0, "rank_biserial": rb}
    cc = 0.5
    z = (W - mean_w + cc) / math.sqrt(var_w) if W < mean_w else (W - mean_w - cc) / math.sqrt(var_w)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    rb = (w_pos - w_neg) / (w_pos + w_neg)
    return {"n": n, "p_value": min(1.0, max(0.0, p)), "rank_biserial": rb}


def holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-topic", type=int, default=2, help="prompts per topic (15 topics)")
    ap.add_argument("--samples-per-segment", type=int, default=3)
    ap.add_argument("--max-sentences", type=int, default=3)
    ap.add_argument("--num-first-candidates", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--limit", type=int, default=0, help="cap prompts per model (0 = no cap; smoke tests)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing defense_samples.jsonl")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip reconstruction; rebuild tables from defense_samples.jsonl")
    args = ap.parse_args()

    per_sample_path = OUT / "defense_samples.jsonl"

    if args.aggregate_only:
        if not per_sample_path.exists():
            raise SystemExit(f"missing {per_sample_path}")
        agg = defaultdict(lambda: {"phi": [], "ed": [], "r1": [], "frames": []})
        per_sample_phi: dict[tuple[str, str, int], float] = {}
        with per_sample_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                key = (o["label"], o["defense"])
                agg[key]["phi"].append(float(o["phi"]))
                agg[key]["ed"].append(float(o["ed_norm"]))
                agg[key]["r1"].append(float(o["r1"]))
                agg[key]["frames"].append(float(o["n_frames"]))
                per_sample_phi[(o["label"], o["defense"], int(o["idx"]))] = float(o["phi"])
        total_done = sum(len(v["phi"]) for v in agg.values())
        t_start = time.time()
    else:
        import torch
        from weiss_reconstruction import reconstruct

        completed = set() if args.fresh else load_completed(per_sample_path)
        mode = "a" if completed and not args.fresh else "w"
        per_sample_f = per_sample_path.open(mode, encoding="utf-8")
        if completed:
            print(f"[defense] resuming; {len(completed)} (model,idx,defense) triples already done", flush=True)

        t_start = time.time()
        total_done = 0
        for label, pretty, run_dir in DEFENSE_RUNS:
            rows = load_samples(run_dir)
            subset = balanced_subset(rows, per_topic=args.per_topic, seed=args.seed)
            if args.limit > 0:
                subset = subset[:args.limit]
            print(f"[defense] {pretty}: {len(subset)} prompts", flush=True)
            for r in subset:
                toks = token_lengths_for(r["run_id"])
                if not toks:
                    continue
                gt = r["response_text"]
                for dname, fn in DEFENSES.items():
                    if (label, r["idx"], dname) in completed:
                        continue
                    seed = (args.seed + r["idx"] * 131 + hash(dname) % 9973) & 0x7fffffff
                    torch.manual_seed(seed)
                    np.random.seed(seed % (2**32 - 1))
                    rng = np.random.default_rng(seed)
                    defended = fn(toks, rng)
                    rec = reconstruct(
                        defended,
                        num_first_candidates=args.num_first_candidates,
                        max_sentences=args.max_sentences,
                        samples_per_segment=args.samples_per_segment,
                    )
                    pred = rec.full_text
                    phi = tf_cosine(gt, pred)
                    ed = levenshtein_norm(gt, pred)
                    r1 = rouge1_precision(gt, pred)
                    per_sample_f.write(json.dumps({
                        "model": pretty, "label": label, "idx": r["idx"], "topic": r["topic"],
                        "defense": dname, "phi": phi, "ed_norm": ed, "r1": r1,
                        "n_frames": len(defended), "pred": pred,
                    }, ensure_ascii=False) + "\n")
                    per_sample_f.flush()
                    total_done += 1
                elapsed = time.time() - t_start
                print(f"[defense] idx={r['idx']:>3} topic={r['topic']:<16} "
                      f"done={total_done} elapsed={elapsed:6.0f}s", flush=True)
        per_sample_f.close()

        agg = defaultdict(lambda: {"phi": [], "ed": [], "r1": [], "frames": []})
        per_sample_phi = {}
        with per_sample_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                key = (o["label"], o["defense"])
                agg[key]["phi"].append(float(o["phi"]))
                agg[key]["ed"].append(float(o["ed_norm"]))
                agg[key]["r1"].append(float(o["r1"]))
                agg[key]["frames"].append(float(o["n_frames"]))
                per_sample_phi[(o["label"], o["defense"], int(o["idx"]))] = float(o["phi"])

    # Aggregate with bootstrap CIs, Wilson LR@0.5, byte overhead, leakage-reduction.
    rng = np.random.default_rng(args.seed)

    def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
        if n == 0:
            return 0.0, 0.0, 0.0
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return p, max(0.0, center - half), min(1.0, center + half)

    # Replay transforms for byte overhead using samples.jsonl run_ids.
    samples_meta: dict[str, dict[int, dict]] = {}
    for label, pretty, run_dir in DEFENSE_RUNS:
        m = {}
        with (RESULTS / run_dir / "samples.jsonl").open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                m[int(o["idx"])] = o
        samples_meta[label] = m

    # Load full per-sample rows for overhead
    sample_rows: list[dict] = []
    with per_sample_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample_rows.append(json.loads(line))

    summary = []
    for label, pretty, _ in DEFENSE_RUNS:
        base_phi = np.array(agg[(label, "baseline")]["phi"]) if agg[(label, "baseline")]["phi"] else np.array([0.0])
        base_mean = float(base_phi.mean())
        for dname in DEFENSES:
            phi = np.array(agg[(label, dname)]["phi"])
            ed = np.array(agg[(label, dname)]["ed"])
            r1 = np.array(agg[(label, dname)]["r1"])
            frames = np.array(agg[(label, dname)]["frames"])
            if len(phi) == 0:
                continue
            n = len(phi)
            bidx = rng.integers(0, n, size=(10000, n))
            bmeans = phi[bidx].mean(axis=1)
            lo, hi = float(np.percentile(bmeans, 2.5)), float(np.percentile(bmeans, 97.5))
            k_succ = int((phi > 0.5).sum())
            lr, lr_lo, lr_hi = wilson(k_succ, n)
            # byte overhead
            overs = []
            for row in sample_rows:
                if row["label"] != label or row["defense"] != dname:
                    continue
                meta = samples_meta[label].get(int(row["idx"]))
                if not meta:
                    continue
                toks = token_lengths_for(meta["run_id"])
                if not toks:
                    continue
                seed = (args.seed + int(row["idx"]) * 131 + hash(dname) % 9973) & 0x7fffffff
                defended = DEFENSES[dname](toks, np.random.default_rng(seed))
                bb, bd = sum(toks), sum(defended)
                overs.append(100.0 * (bd - bb) / bb if bb else 0.0)
            red = 100.0 * (1 - phi.mean() / base_mean) if base_mean > 0 else 0.0
            summary.append({
                "model": pretty, "label": label, "defense": dname, "n": n,
                "phi_mean": float(phi.mean()), "phi_ci": [lo, hi],
                "ed_norm_mean": float(ed.mean()), "r1_mean": float(r1.mean()),
                "lr_at_0_5_pct": 100 * lr, "lr_ci": [100 * lr_lo, 100 * lr_hi],
                "asr": 100 * lr,  # back-compat
                "mean_frames": float(frames.mean()),
                "byte_overhead_pct_mean": float(np.mean(overs)) if overs else float("nan"),
                "leakage_reduction_pct": red,
            })
    (OUT / "defense_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # LaTeX: Model & Defense & n & phi & CI & LR@0.5 [Wilson] & byte OH% & leak red
    esc = lambda s: s.replace("_", "\\_")
    lines = ["% Auto-generated by defense_eval.py -- defense table body"]
    prev_model = None
    for s in summary:
        model_cell = s["model"] if s["model"] != prev_model else ""
        if s["model"] != prev_model and prev_model is not None:
            lines.append("\\midrule")
        prev_model = s["model"]
        lines.append(
            f"{model_cell} & {esc(s['defense'])} & {s['n']} & "
            f"{s['phi_mean']:.4f} & {{[{s['phi_ci'][0]:.4f}, {s['phi_ci'][1]:.4f}]}} & "
            f"{s['lr_at_0_5_pct']:.1f} {{[{s['lr_ci'][0]:.1f}, {s['lr_ci'][1]:.1f}]}} & "
            f"{s['byte_overhead_pct_mean']:.1f} & "
            f"{s['leakage_reduction_pct']:.1f} \\\\")
    (OUT / "defense_tables_body.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Paired Wilcoxon: each defense vs baseline (per model), Holm over 10 tests.
    pairwise = []
    raw_p = []
    for label, pretty, _ in DEFENSE_RUNS:
        base_by_idx = {
            idx: per_sample_phi[(label, "baseline", idx)]
            for idx in sorted(k[2] for k in per_sample_phi if k[0] == label and k[1] == "baseline")
        }
        if not base_by_idx:
            continue
        idxs = sorted(base_by_idx)
        base = np.array([base_by_idx[i] for i in idxs])
        for dname in DEFENSES:
            if dname == "baseline":
                continue
            def_by_idx = {
                idx: per_sample_phi[(label, dname, idx)]
                for idx in idxs
                if (label, dname, idx) in per_sample_phi
            }
            if len(def_by_idx) != len(idxs):
                continue
            defended = np.array([def_by_idx[i] for i in idxs])
            w = wilcoxon_signed_rank(base, defended)
            dmean = float((base - defended).mean())
            pairwise.append({
                "model": pretty, "label": label, "defense": dname, "n": len(idxs),
                "mean_phi_baseline": float(base.mean()),
                "mean_phi_defended": float(defended.mean()),
                "mean_reduction": dmean,
                "p_raw": w["p_value"], "rank_biserial": w["rank_biserial"],
            })
            raw_p.append(w["p_value"])
    if raw_p:
        adj = holm_bonferroni(raw_p)
        for pinfo, pa in zip(pairwise, adj):
            pinfo["p_holm"] = pa
            pinfo["significant_0_05"] = pa < 0.05
        (OUT / "defense_pairwise_tests.json").write_text(
            json.dumps(pairwise, indent=2), encoding="utf-8")
        pw_lines = ["% Auto-generated by defense_eval.py -- defense vs baseline Wilcoxon"]
        for p in sorted(pairwise, key=lambda x: x["p_holm"]):
            star = "$^{*}$" if p["significant_0_05"] else ""
            esc = lambda s: s.replace("_", "\\_")
            pstr = "<0.0001" if p["p_holm"] < 1e-4 else f"{p['p_holm']:.4f}"
            pw_lines.append(
                f"{p['model']} & {esc(p['defense'])} & {p['n']} & "
                f"{p['mean_phi_baseline']:.4f} & {p['mean_phi_defended']:.4f} & "
                f"{p['mean_reduction']:+.4f} & {p['rank_biserial']:+.3f} & {pstr}{star} \\\\")
        (OUT / "defense_pairwise.tex").write_text("\n".join(pw_lines) + "\n", encoding="utf-8")

    print(f"[defense] DONE total={total_done} in {time.time()-t_start:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
