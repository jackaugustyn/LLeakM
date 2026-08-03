"""Statistical analysis for the SSE frame-length reconstruction study.

This script produces the quantitative evidence required for a rigorous
comparative evaluation:

  * A single, fixed semantic backend (term-frequency cosine) recomputed from
    the stored ground-truth / reconstruction text of every sample, so that phi
    is comparable across all runs (the original campaign mixed a MiniLM backend
    for one run with the TF-IDF backend for the others).
  * Bootstrap 95% confidence intervals for mean semantic similarity, mean
    normalized edit distance, ROUGE precision, and the attack success rate.
  * Paired Wilcoxon signed-rank tests between every pair of victim models
    (prompts are aligned by ``idx`` across runs), with Holm-Bonferroni
    correction for multiple comparisons and a rank-biserial effect size.

All randomness is seeded, and the script depends only on the standard library
plus NumPy so it runs in the frozen project environment without SciPy/sklearn.

Outputs (written to ``experiment_validation/analysis/``):
  * ``model_stats.json``       -- per-model point estimates and CIs
  * ``pairwise_tests.json``    -- paired Wilcoxon results with correction
  * ``per_prompt_phi.csv``     -- aligned per-prompt phi matrix (for figures)
  * ``tables.tex``             -- ready-to-include LaTeX table bodies
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment_validation" / "results"
OUT = ROOT / "experiment_validation" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260706
N_BOOT = 10000
ASR_THRESHOLD = 0.5

# Standardized comparison subset: N=300, max_new_tokens=96, sps=3, max_sent=3.
# (label, pretty name, run directory)
RUNS = [
    ("qwen_1_5b", "Qwen2.5-1.5B-Instruct", "run_20260325_104518_qwen_1_5b"),
    ("qwen_3b", "Qwen2.5-3B-Instruct", "run_20260327_124817_qwen_3b_final_resume"),
    ("llama_3_2_3b", "Llama-3.2-3B-Instruct", "run_20260414_073030_llama_3_2_3b"),
    ("gemma_2_2b", "Gemma-2-2B-it", "run_20260415_080034_gemma_2_2b_it"),
    ("phi_1_5", "Phi-1.5", "run_20260329_085714_phi_1_5"),
    ("tinyllama_1_1b", "TinyLlama-1.1B-Chat", "run_20260325_153010_tinyllama_1_1b"),
    ("phi_3_5_mini", "Phi-3.5-mini-instruct", "run_20260414_111644_phi_3_5_mini"),
]

WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def tf_cosine(a: str, b: str) -> float:
    """Term-frequency cosine similarity, matching the harness TF-IDF fallback.

    This is the single fixed semantic backend used for every sample so that phi
    is measured identically across all victim models.
    """
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
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def load_run(run_dir: str) -> dict[int, dict]:
    """Return {idx: sample dict} with a recomputed, backend-consistent phi."""
    path = RESULTS / run_dir / "samples.jsonl"
    rows: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            phi = tf_cosine(o.get("response_text", ""), o.get("pred_full_text", ""))
            rows[o["idx"]] = {
                "idx": o["idx"],
                "topic": o["topic"],
                "phi": phi,
                "ed_norm": float(o["ed_norm"]),
                "r1": float(o["rouge1_precision"]),
                "rl": float(o["rougeL_precision"]),
            }
    return rows


def bootstrap_ci_mean(values: np.ndarray, rng: np.random.Generator,
                      n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson score interval for a proportion (ASR)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> dict:
    """Two-sided paired Wilcoxon signed-rank test (normal approximation).

    Uses average ranks for ties, the standard tie correction on the variance,
    and a continuity correction. Zero differences are dropped (Wilcoxon
    exclusion). Returns the statistic, z, p-value, and the rank-biserial
    effect size r = (W+ - W-) / (W+ + W-).
    """
    d = x - y
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": 0.0, "z": 0.0, "p_value": 1.0, "rank_biserial": 0.0}
    ranks = _average_ranks(np.abs(d))
    w_pos = float(ranks[d > 0].sum())
    w_neg = float(ranks[d < 0].sum())
    W = min(w_pos, w_neg)
    mean_w = n * (n + 1) / 4.0
    # Tie correction for variance.
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts))
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return {"n": n, "W": W, "z": 0.0, "p_value": 1.0,
                "rank_biserial": (w_pos - w_neg) / (w_pos + w_neg)}
    cc = 0.5  # continuity correction
    z = (W - mean_w + cc) / math.sqrt(var_w) if W < mean_w else (W - mean_w - cc) / math.sqrt(var_w)
    p = 2.0 * (1.0 - _std_normal_cdf(abs(z)))
    p = min(1.0, max(0.0, p))
    rb = (w_pos - w_neg) / (w_pos + w_neg)
    return {"n": n, "W": W, "z": z, "p_value": p, "rank_biserial": rb}


def _average_ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # ranks are 1-based
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _std_normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values preserving input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def paired_diff_bootstrap_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                             n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    d = x - y
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def main() -> None:
    rng = np.random.default_rng(SEED)

    data = {label: load_run(run_dir) for label, _, run_dir in RUNS}
    # Common prompt indices across all runs (paired design).
    common = set.intersection(*[set(d.keys()) for d in data.values()])
    common_sorted = sorted(common)
    print(f"[stats] common aligned prompts across all runs: {len(common_sorted)}")

    # -------- Per-model point estimates + CIs --------
    model_stats = []
    phi_by_model: dict[str, np.ndarray] = {}
    for label, pretty, run_dir in RUNS:
        rows = data[label]
        phi = np.array([rows[i]["phi"] for i in common_sorted])
        ed = np.array([rows[i]["ed_norm"] for i in common_sorted])
        r1 = np.array([rows[i]["r1"] for i in common_sorted])
        rl = np.array([rows[i]["rl"] for i in common_sorted])
        phi_by_model[label] = phi

        phi_m, phi_lo, phi_hi = bootstrap_ci_mean(phi, rng)
        ed_m, ed_lo, ed_hi = bootstrap_ci_mean(ed, rng)
        r1_m, r1_lo, r1_hi = bootstrap_ci_mean(r1, rng)
        rl_m, rl_lo, rl_hi = bootstrap_ci_mean(rl, rng)
        k = int((phi > ASR_THRESHOLD).sum())
        asr_p, asr_lo, asr_hi = wilson_ci(k, len(phi))

        model_stats.append({
            "label": label, "model": pretty, "run_dir": run_dir, "n": len(phi),
            "phi_mean": phi_m, "phi_ci": [phi_lo, phi_hi],
            "ed_norm_mean": ed_m, "ed_norm_ci": [ed_lo, ed_hi],
            "r1_mean": r1_m, "r1_ci": [r1_lo, r1_hi],
            "rl_mean": rl_m, "rl_ci": [rl_lo, rl_hi],
            "asr": 100 * asr_p, "asr_ci": [100 * asr_lo, 100 * asr_hi],
            "asr_success": k,
        })
        print(f"[stats] {pretty:26s} phi={phi_m:.4f} [{phi_lo:.4f},{phi_hi:.4f}] "
              f"ASR={100*asr_p:.2f}% [{100*asr_lo:.2f},{100*asr_hi:.2f}]")

    # -------- Pairwise paired Wilcoxon tests on phi --------
    labels = [r[0] for r in RUNS]
    pretty = {r[0]: r[1] for r in RUNS}
    pairs = []
    raw_p = []
    for a_i in range(len(labels)):
        for b_i in range(a_i + 1, len(labels)):
            la, lb = labels[a_i], labels[b_i]
            x, y = phi_by_model[la], phi_by_model[lb]
            w = wilcoxon_signed_rank(x, y)
            dmean, dlo, dhi = paired_diff_bootstrap_ci(x, y, rng)
            pairs.append({
                "a": la, "b": lb, "a_model": pretty[la], "b_model": pretty[lb],
                "mean_diff_phi": dmean, "diff_ci": [dlo, dhi],
                "z": w["z"], "p_raw": w["p_value"], "rank_biserial": w["rank_biserial"],
                "n_nonzero": w["n"],
            })
            raw_p.append(w["p_value"])
    adj = holm_bonferroni(raw_p)
    for pinfo, pa in zip(pairs, adj):
        pinfo["p_holm"] = pa
        pinfo["significant_0_05"] = pa < 0.05

    # -------- Persist --------
    (OUT / "model_stats.json").write_text(json.dumps(model_stats, indent=2), encoding="utf-8")
    (OUT / "pairwise_tests.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    with (OUT / "per_prompt_phi.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["idx", "topic"] + labels)
        for i in common_sorted:
            topic = data[labels[0]][i]["topic"]
            wtr.writerow([i, topic] + [f"{data[l][i]['phi']:.6f}" for l in labels])

    _write_tables(model_stats, pairs)
    _write_topic_table(data, common_sorted)
    print(f"[stats] wrote outputs to {OUT}")


def _write_tables(model_stats: list[dict], pairs: list[dict]) -> None:
    # Main per-model table body (rows only; sorted by descending phi).
    main = ["% Auto-generated by analyze_stats.py -- main per-model table body"]
    for m in sorted(model_stats, key=lambda x: x["phi_mean"], reverse=True):
        main.append(
            f"{m['model']} & {m['phi_mean']:.4f} & "
            f"{{[{m['phi_ci'][0]:.4f}, {m['phi_ci'][1]:.4f}]}} & "
            f"{m['ed_norm_mean']:.4f} & {m['r1_mean']:.4f} & {m['rl_mean']:.4f} & "
            f"{m['asr']:.2f} {{[{m['asr_ci'][0]:.2f}, {m['asr_ci'][1]:.2f}]}} \\\\")
    (OUT / "tables_main.tex").write_text("\n".join(main) + "\n", encoding="utf-8")

    # Pairwise Wilcoxon table body (most significant first).
    pw = ["% Auto-generated by analyze_stats.py -- pairwise Wilcoxon table body"]
    ordered = sorted(pairs, key=lambda p: p["p_holm"])
    for p in ordered:
        star = "$^{*}$" if p["significant_0_05"] else ""
        pw.append(
            f"{p['a_model']} vs.\\ {p['b_model']} & "
            f"{p['mean_diff_phi']:+.4f} & {{[{p['diff_ci'][0]:+.4f}, {p['diff_ci'][1]:+.4f}]}} & "
            f"{p['rank_biserial']:+.3f} & {_fmt_p(p['p_raw'])} & {_fmt_p(p['p_holm'])}{star} \\\\")
    (OUT / "tables_pairwise.tex").write_text("\n".join(pw) + "\n", encoding="utf-8")

    # Keep a combined copy for convenience.
    (OUT / "tables.tex").write_text("\n".join(main) + "\n\n" + "\n".join(pw) + "\n", encoding="utf-8")


TOPIC_MODELS = [
    ("llama_3_2_3b", "Llama-3.2-3B-Instruct"),
    ("gemma_2_2b", "Gemma-2-2B-it"),
    ("phi_3_5_mini", "Phi-3.5-mini-instruct"),
]


def _write_topic_table(data: dict[str, dict[int, dict]], common: list[int]) -> None:
    """Best/worst topic per selected model using recomputed phi."""
    lines = ["% Auto-generated by analyze_stats.py -- topic sensitivity (best/worst)"]
    for label, pretty in TOPIC_MODELS:
        by_topic: dict[str, list[float]] = {}
        for idx in common:
            row = data[label][idx]
            by_topic.setdefault(row["topic"], []).append(row["phi"])
        topic_mean = {t: float(np.mean(v)) for t, v in by_topic.items()}
        best_t = max(topic_mean, key=topic_mean.get)
        worst_t = min(topic_mean, key=topic_mean.get)
        lines.append(
            f"{pretty} & {best_t.replace('_', '\\_')} & {topic_mean[best_t]:.4f} & "
            f"{worst_t.replace('_', '\\_')} & {topic_mean[worst_t]:.4f} \\\\")
    (OUT / "tables_topic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return "$<$0.0001"
    return f"{p:.4f}"


if __name__ == "__main__":
    main()
