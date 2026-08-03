"""Extra analyses for the revised manuscript.

Produces:
  * byte-overhead and Wilson CI for defenses (replaying transforms; no T5)
  * instruct-only main table (excludes Phi-1.5)
  * OLS-style linear model of phi ~ model + covariates (NumPy only)
  * qualitative reconstruction examples
  * model/tokenizer metadata table
  * topic heatmap body (exploratory appendix)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment_validation" / "results"
LOGDIR = ROOT / "logs"
OUT = ROOT / "experiment_validation" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"[A-Za-z0-9']+")
SEED = 20260706
ASR_THRESHOLD = 0.5

RUNS = [
    ("qwen_1_5b", "Qwen2.5-1.5B-Instruct", "run_20260325_104518_qwen_1_5b",
     "Qwen/Qwen2.5-1.5B-Instruct", "instruct", "~1.5B", "Qwen2Tokenizer"),
    ("qwen_3b", "Qwen2.5-3B-Instruct", "run_20260327_124817_qwen_3b_final_resume",
     "Qwen/Qwen2.5-3B-Instruct", "instruct", "~3B", "Qwen2Tokenizer"),
    ("llama_3_2_3b", "Llama-3.2-3B-Instruct", "run_20260414_073030_llama_3_2_3b",
     "meta-llama/Llama-3.2-3B-Instruct", "instruct", "~3B", "LlamaTokenizer"),
    ("gemma_2_2b", "Gemma-2-2B-it", "run_20260415_080034_gemma_2_2b_it",
     "google/gemma-2-2b-it", "instruct", "~2B", "GemmaTokenizer"),
    ("phi_1_5", "Phi-1.5", "run_20260329_085714_phi_1_5",
     "microsoft/phi-1_5", "base", "~1.3B", "CodeGenTokenizer"),
    ("tinyllama_1_1b", "TinyLlama-1.1B-Chat", "run_20260325_153010_tinyllama_1_1b",
     "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "chat", "~1.1B", "LlamaTokenizer"),
    ("phi_3_5_mini", "Phi-3.5-mini-instruct", "run_20260414_111644_phi_3_5_mini",
     "microsoft/Phi-3.5-mini-instruct", "instruct", "~3.8B", "LlamaTokenizer"),
]

INSTRUCT_LABELS = {r[0] for r in RUNS if r[4] != "base"}

DEFENSE_RUNS = [
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


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def token_lengths_for(run_id: str) -> list[int] | None:
    p = LOGDIR / f"{run_id}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return [int(s["token_utf8_len"]) for s in d.get("steps", [])]


# ---- online-safe defense transforms (must match defense_eval.py) ----

def d_baseline(t: list[int], rng: np.random.Generator) -> list[int]:
    return list(t)


def d_bucket8(t: list[int], rng: np.random.Generator) -> list[int]:
    return [int(math.ceil(x / 8.0) * 8) for x in t]


def d_pad32(t: list[int], rng: np.random.Generator) -> list[int]:
    """Online fixed frame size C=32; tokens longer than C still emit size C."""
    return [32 for _ in t]


def d_batch2(t: list[int], rng: np.random.Generator) -> list[int]:
    return [int(sum(t[i:i + 2])) for i in range(0, len(t), 2)]


def d_batch4(t: list[int], rng: np.random.Generator) -> list[int]:
    return [int(sum(t[i:i + 4])) for i in range(0, len(t), 4)]


def d_rand_pad8(t: list[int], rng: np.random.Generator) -> list[int]:
    return [int(x + rng.integers(0, 9)) for x in t]


DEFENSE_FNS = {
    "baseline": d_baseline,
    "bucket_8": d_bucket8,
    "pad_32": d_pad32,
    "batch_2": d_batch2,
    "batch_4": d_batch4,
    "rand_pad_8": d_rand_pad8,
}

# Map legacy name in jsonl to online pad_32 for overhead display when needed
LEGACY_PAD = "pad_fixed"


def bytes_of(lengths: list[int]) -> int:
    # Observable payload proxy: sum of emitted frame body lengths (token channel).
    return int(sum(lengths))


def defense_overhead_and_wilson() -> None:
    """Replay transforms for byte overhead; Wilson CI from defense_samples.jsonl."""
    path = OUT / "defense_samples.jsonl"
    if not path.exists():
        print("[extras] no defense_samples.jsonl")
        return

    # Load per-sample phi by (label, defense, idx); also need run_id via samples
    rows_by_run: dict[str, dict[int, dict]] = {}
    for label, pretty, run_dir in DEFENSE_RUNS:
        m = {}
        with (RESULTS / run_dir / "samples.jsonl").open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                m[int(o["idx"])] = o
        rows_by_run[label] = m

    per = defaultdict(list)  # (label, defense) -> list of dicts
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            dname = o["defense"]
            if dname == LEGACY_PAD:
                dname = "pad_32_legacy_oracle"  # keep separate; overhead uses online pad_32 below
            per[(o["label"], o["defense"])].append(o)

    rng = np.random.default_rng(SEED)
    summary = []
    # Prefer canonical defense names for table; for pad_fixed rows, also compute online pad_32 overhead
    # from the same prompt indices while reporting legacy phi under note in paper if pad_32 missing.
    for label, pretty, _ in DEFENSE_RUNS:
        base_rows = per.get((label, "baseline"), [])
        if not base_rows:
            continue
        # byte baseline per idx
        base_bytes = {}
        for r in base_rows:
            sample = rows_by_run[label].get(int(r["idx"]))
            if not sample:
                continue
            toks = token_lengths_for(sample["run_id"])
            if not toks:
                continue
            base_bytes[int(r["idx"])] = bytes_of(toks)

        for dname in ["baseline", "bucket_8", "pad_fixed", "pad_32", "batch_2", "batch_4", "rand_pad_8"]:
            rows = per.get((label, dname), [])
            if not rows and dname == "pad_32":
                # compute overhead-only from baseline indices using online pad_32
                continue
            if not rows:
                continue
            phis = np.array([float(r["phi"]) for r in rows])
            n = len(phis)
            k = int((phis > ASR_THRESHOLD).sum())
            lr, lr_lo, lr_hi = wilson_ci(k, n)
            idx = rng.integers(0, n, size=(10000, n))
            blo, bhi = float(np.percentile(phis[idx].mean(axis=1), 2.5)), float(
                np.percentile(phis[idx].mean(axis=1), 97.5))

            overs = []
            frames = []
            for r in rows:
                i = int(r["idx"])
                sample = rows_by_run[label].get(i)
                if not sample or i not in base_bytes:
                    continue
                toks = token_lengths_for(sample["run_id"])
                if not toks:
                    continue
                seed = (SEED + i * 131 + hash(dname if dname != "pad_fixed" else "pad_32") % 9973) & 0x7fffffff
                rr = np.random.default_rng(seed)
                fn_name = "pad_32" if dname == "pad_fixed" else dname
                # For overhead of pad_fixed legacy oracle vs online: report both
                if dname == "pad_fixed":
                    # legacy oracle overhead
                    c = max(max(toks), 16)
                    defended = [c for _ in toks]
                else:
                    defended = DEFENSE_FNS[fn_name](toks, rr)
                bb = base_bytes[i]
                bd = bytes_of(defended)
                overs.append(100.0 * (bd - bb) / bb if bb > 0 else 0.0)
                frames.append(len(defended))

            base_phi = np.array([float(r["phi"]) for r in base_rows])
            red = 100.0 * (1 - phis.mean() / base_phi.mean()) if base_phi.mean() > 0 else 0.0
            display = "pad_32*" if dname == "pad_fixed" else dname  # * = oracle max-L (legacy)
            summary.append({
                "model": pretty, "label": label, "defense": dname, "display": display,
                "n": n, "phi_mean": float(phis.mean()), "phi_ci": [blo, bhi],
                "lr_at_0_5_pct": 100 * lr, "lr_ci": [100 * lr_lo, 100 * lr_hi],
                "byte_overhead_pct_mean": float(np.mean(overs)) if overs else float("nan"),
                "mean_frames": float(np.mean(frames)) if frames else float("nan"),
                "leakage_reduction_pct": red,
            })

        # Always add online pad_32 overhead row even without reconstructions yet
        overs32, frames32 = [], []
        for i, bb in base_bytes.items():
            sample = rows_by_run[label][i]
            toks = token_lengths_for(sample["run_id"])
            if not toks:
                continue
            defended = d_pad32(toks, np.random.default_rng(0))
            overs32.append(100.0 * (bytes_of(defended) - bb) / bb)
            frames32.append(len(defended))
        if overs32:
            summary.append({
                "model": pretty, "label": label, "defense": "pad_32_overhead_only",
                "display": "pad_32",
                "n": 0, "phi_mean": None, "phi_ci": None,
                "lr_at_0_5_pct": None, "lr_ci": None,
                "byte_overhead_pct_mean": float(np.mean(overs32)),
                "mean_frames": float(np.mean(frames32)),
                "leakage_reduction_pct": None,
                "note": "online C=32; reconstruction pending if absent",
            })

    (OUT / "defense_overhead.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Compact LaTeX: Model, Defense, n, phi, LR@0.5 [Wilson], byte OH%, leak red
    lines = ["% Auto-generated -- defense metrics with byte overhead and Wilson LR@0.5"]
    prev = None
    for s in summary:
        if s["defense"] in ("pad_32_overhead_only",):
            continue
        if s["defense"] == "pad_fixed":
            dshow = "pad\\_32$^{\\dagger}$"
        else:
            dshow = s["defense"].replace("_", "\\_")
        if s["model"] != prev and prev is not None:
            lines.append("\\midrule")
        model_cell = s["model"] if s["model"] != prev else ""
        prev = s["model"]
        lr = s["lr_at_0_5_pct"]
        lci = s["lr_ci"]
        oh = s["byte_overhead_pct_mean"]
        lines.append(
            f"{model_cell} & {dshow} & {s['n']} & {s['phi_mean']:.4f} & "
            f"{lr:.1f} {{[{lci[0]:.1f}, {lci[1]:.1f}]}} & "
            f"{oh:.1f} & {s['leakage_reduction_pct']:.1f} \\\\")
    (OUT / "defense_tables_overhead.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[extras] wrote defense_overhead.json ({len(summary)} rows)")


def load_recomputed_phi() -> dict[str, dict[int, float]]:
    """Recompute TF cosine phi from stored texts (consistent backend)."""
    out: dict[str, dict[int, float]] = {}
    for label, _, run_dir, *_ in RUNS:
        m = {}
        with (RESULTS / run_dir / "samples.jsonl").open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                m[int(o["idx"])] = tf_cosine(o.get("response_text", ""), o.get("pred_full_text", ""))
        out[label] = m
    return out


def covariates_for(label: str, run_dir: str) -> dict[int, dict]:
    rows = {}
    with (RESULTS / run_dir / "samples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            idx = int(o["idx"])
            nt = int(o.get("token_count") or 0)
            chars = len(o.get("response_text", ""))
            toks = token_lengths_for(o.get("run_id", ""))
            mean_tb = float(np.mean(toks)) if toks else (chars / max(nt, 1))
            rows[idx] = {
                "chars": chars, "n_tokens": nt if nt else (len(toks) if toks else 0),
                "mean_token_bytes": mean_tb,
                "truncated": 1.0 if (nt >= 96 or (toks and len(toks) >= 96)) else 0.0,
                "topic": o.get("topic", ""),
            }
    return rows


def ols_covariate_model(phi_by: dict[str, dict[int, float]]) -> None:
    """phi ~ model dummies + chars + mean_token_bytes + truncated (instruct models only)."""
    labels = [r[0] for r in RUNS if r[0] in INSTRUCT_LABELS]
    run_map = {r[0]: r[2] for r in RUNS}
    pretty = {r[0]: r[1] for r in RUNS}

    rows = []
    for lab in labels:
        cov = covariates_for(lab, run_map[lab])
        for idx, phi in phi_by[lab].items():
            if idx not in cov:
                continue
            c = cov[idx]
            rows.append({
                "phi": phi, "label": lab,
                "chars": c["chars"], "mean_token_bytes": c["mean_token_bytes"],
                "truncated": c["truncated"],
            })

    # Design matrix: intercept + model dummies (drop first) + covariates
    n = len(rows)
    k_models = len(labels) - 1
    X = np.zeros((n, 1 + k_models + 3))
    y = np.zeros(n)
    X[:, 0] = 1.0
    for i, r in enumerate(rows):
        y[i] = r["phi"]
        mi = labels.index(r["label"])
        if mi > 0:
            X[i, mi] = 1.0  # columns 1..k_models
        X[i, 1 + k_models] = r["chars"] / 100.0  # scale
        X[i, 2 + k_models] = r["mean_token_bytes"]
        X[i, 3 + k_models] = r["truncated"]

    # Least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Compare to model-only and covariate-only
    X_m = X[:, : 1 + k_models]
    beta_m, *_ = np.linalg.lstsq(X_m, y, rcond=None)
    r2_m = 1 - float(np.sum((y - X_m @ beta_m) ** 2)) / ss_tot
    X_c = np.column_stack([np.ones(n), X[:, 1 + k_models:]])
    beta_c, *_ = np.linalg.lstsq(X_c, y, rcond=None)
    r2_c = 1 - float(np.sum((y - X_c @ beta_c) ** 2)) / ss_tot

    coefs = {"intercept": float(beta[0])}
    for j, lab in enumerate(labels[1:]):
        coefs[f"model:{pretty[lab]}"] = float(beta[1 + j])
    coefs["chars/100"] = float(beta[1 + k_models])
    coefs["mean_token_bytes"] = float(beta[2 + k_models])
    coefs["truncated"] = float(beta[3 + k_models])

    out = {
        "n": n, "r2_full": r2, "r2_model_only": r2_m, "r2_covariates_only": r2_c,
        "reference_model": pretty[labels[0]], "coefficients": coefs,
    }
    (OUT / "covariate_regression.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "% Auto-generated covariate regression summary",
        f"Full model $R^2$ & {r2:.3f} \\\\",
        f"Model indicators only $R^2$ & {r2_m:.3f} \\\\",
        f"Covariates only (chars, token bytes, trunc.) $R^2$ & {r2_c:.3f} \\\\",
        f"$\\beta_{{\\mathrm{{chars}}/100}}$ & {coefs['chars/100']:+.4f} \\\\",
        f"$\\beta_{{\\mathrm{{mean\\,token\\,bytes}}}}$ & {coefs['mean_token_bytes']:+.4f} \\\\",
        f"$\\beta_{{\\mathrm{{truncated}}}}$ & {coefs['truncated']:+.4f} \\\\",
    ]
    (OUT / "tables_regression.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[extras] regression R2_full={r2:.3f} R2_model={r2_m:.3f} R2_cov={r2_c:.3f}")


def instruct_only_table(phi_by: dict[str, dict[int, float]]) -> None:
    stats = json.loads((OUT / "model_stats.json").read_text())
    lines = ["% Instruct/chat models only (Phi-1.5 excluded)"]
    for m in sorted(stats, key=lambda x: x["phi_mean"], reverse=True):
        if m["label"] not in INSTRUCT_LABELS:
            continue
        lines.append(
            f"{m['model']} & {m['phi_mean']:.4f} & "
            f"{{[{m['phi_ci'][0]:.4f}, {m['phi_ci'][1]:.4f}]}} & "
            f"{m['ed_norm_mean']:.4f} & {m['r1_mean']:.4f} & "
            f"{m['asr']:.2f} {{[{m['asr_ci'][0]:.2f}, {m['asr_ci'][1]:.2f}]}} \\\\")
    (OUT / "tables_main_instruct.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Pairwise among instruct only
    pairs = json.loads((OUT / "pairwise_tests.json").read_text())
    pw = ["% Instruct-only pairwise (subset of full Holm table; p_Holm from full 21-test correction)"]
    for p in sorted(pairs, key=lambda x: x["p_holm"]):
        if p["a"] not in INSTRUCT_LABELS or p["b"] not in INSTRUCT_LABELS:
            continue
        star = "$^{*}$" if p["significant_0_05"] else ""
        pstr = "$<$0.0001" if p["p_holm"] < 1e-4 else f"{p['p_holm']:.4f}"
        pw.append(
            f"{p['a_model']} vs.\\ {p['b_model']} & "
            f"{p['mean_diff_phi']:+.4f} & {{[{p['diff_ci'][0]:+.4f}, {p['diff_ci'][1]:+.4f}]}} & "
            f"{p['rank_biserial']:+.3f} & {pstr}{star} \\\\")
    (OUT / "tables_pairwise_instruct.tex").write_text("\n".join(pw) + "\n", encoding="utf-8")
    print("[extras] wrote instruct-only tables")


def qualitative_examples(phi_by: dict[str, dict[int, float]]) -> None:
    """Pick high/mid/low phi examples + one defense contrast if available."""
    examples = []
    # From Qwen (strong) and Phi-3.5 (weak)
    for label, pretty, run_dir, *_ in RUNS:
        if label not in ("qwen_1_5b", "phi_3_5_mini", "llama_3_2_3b"):
            continue
        items = []
        with (RESULTS / run_dir / "samples.jsonl").open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                idx = int(o["idx"])
                phi = phi_by[label][idx]
                items.append((phi, o))
        items.sort(key=lambda x: x[0], reverse=True)
        for tag, pick in [("high", items[0]), ("mid", items[len(items) // 2]), ("low", items[-1])]:
            phi, o = pick
            examples.append({
                "model": pretty, "tag": tag, "idx": o["idx"], "topic": o["topic"],
                "phi": phi,
                "response": o["response_text"][:280],
                "pred": o["pred_full_text"][:280],
            })

    (OUT / "qualitative_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    # LaTeX-friendly short table (escaped)
    def esc(s: str) -> str:
        return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
                .replace("_", "\\_").replace("#", "\\#"))

    lines = ["% Qualitative examples (truncated for display)"]
    for e in examples:
        if e["tag"] != "high" and not (e["model"].startswith("Qwen") and e["tag"] == "low"):
            if not (e["model"].startswith("Phi-3.5") and e["tag"] == "high"):
                continue
        lines.append(
            f"{esc(e['model'])} & {e['tag']} & {e['phi']:.3f} & "
            f"\\texttt{{{esc(e['response'][:90])}...}} & "
            f"\\texttt{{{esc(e['pred'][:90])}...}} \\\\")
    (OUT / "tables_qualitative.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[extras] wrote {len(examples)} qualitative examples")


def metadata_table() -> None:
    lines = ["% Model metadata"]
    for _, pretty, _, hf_id, kind, params, tok in RUNS:
        lines.append(
            f"{pretty} & {params} & {kind} & \\texttt{{{hf_id}}} & {tok} \\\\")
    (OUT / "tables_metadata.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Environment snapshot
    import platform
    import sys
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch
        env["torch"] = torch.__version__
    except ImportError:
        env["torch"] = "n/a"
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        env["transformers"] = "n/a"
    (OUT / "env_snapshot.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("[extras] wrote metadata + env snapshot")


def topic_heatmap(phi_by: dict[str, dict[int, float]]) -> None:
    """Mean phi per (model, topic) for appendix heatmap-style table."""
    topics = set()
    data = {}
    for label, pretty, run_dir, *_ in RUNS:
        cov = covariates_for(label, run_dir)
        by_t: dict[str, list[float]] = defaultdict(list)
        for idx, phi in phi_by[label].items():
            t = cov[idx]["topic"]
            by_t[t].append(phi)
            topics.add(t)
        data[pretty] = {t: float(np.mean(v)) for t, v in by_t.items()}
    topics = sorted(topics)
    # Compact: one row per model, columns = abbreviated topics too wide; instead best/worst already exists.
    # Write JSON for reproducibility; LaTeX: mean range only
    (OUT / "topic_matrix.json").write_text(json.dumps({"topics": topics, "means": data}, indent=2), encoding="utf-8")
    print("[extras] wrote topic_matrix.json")


def main() -> None:
    defense_overhead_and_wilson()
    phi_by = load_recomputed_phi()
    ols_covariate_model(phi_by)
    instruct_only_table(phi_by)
    qualitative_examples(phi_by)
    metadata_table()
    topic_heatmap(phi_by)
    print(f"[extras] done -> {OUT}")


if __name__ == "__main__":
    main()
