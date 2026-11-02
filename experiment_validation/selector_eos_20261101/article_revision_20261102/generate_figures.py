#!/usr/bin/env python3
"""Regenerate manuscript plots from completed, independently audited results.

Run from any directory with Python, matplotlib and numpy. No model inference.
Numeric input, CSV exports, source hashes and output hashes accompany the plots.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = HERE.parent / "validation"
OUT = ROOT / "article1" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_NAMES = ["main_fresh_factledger.json", "secondary_audit.json",
                "paired_detokenization_score_contrast.json"]
main, secondary, paired = [json.loads((AUDIT / n).read_text()) for n in SOURCE_NAMES]
assert main["all_checks_passed"] and secondary["status"] == "PASS"
assert paired["complete"] and paired["independent_recalculation_agrees"]
assert paired["selector_version"] == "eos-through-first-inclusive/1"

def score_row(m):
    lo, hi = m["phi_ci_scenario_bootstrap"]
    return dict(model=m["label"], mean=m["metrics"]["phi"]["mean"], low=lo, high=hi)

archived_by_label = {m["label"]: m for m in main["cohorts"]["cuda_full"]["models"]}
order = sorted(archived_by_label,
               key=lambda k: archived_by_label[k]["metrics"]["phi"]["mean"], reverse=True)
data = {"archived_scenario": [score_row(archived_by_label[k]) for k in order],
        "fresh_scenario": [score_row(main["cohorts"][cohort]["models"][0])
                           for cohort in ("fresh_tinyllama_cuda", "fresh_phi_cuda")],
        "paired_contrasts": [dict(model=m["label"], mean_difference=m["delta_phi_mean"],
                                  low=m["delta_phi_ci_paired_scenario"][0],
                                  high=m["delta_phi_ci_paired_scenario"][1])
                             for m in paired["models"]]}
conditions = ["bucket_8", "pad_32", "batch_2", "batch_4", "rand_pad_8"]
for label, key in [("qwen_1_5b", "defense_qwen"), ("llama_3_2_3b", "defense_llama")]:
    cells = {c["condition"]: c for c in secondary["cells"]
             if c["suite"] == "defense_pilot" and c["label"] == label}
    contrasts = {c["condition"]: c for c in secondary["paired_contrasts"]
                 if c["suite"] == "defense_pilot" and c["label"] == label}
    data[key] = [dict(condition=c, overhead=cells[c]["observed_payload_overhead_pct_mean"],
                      reduction=contrasts[c]["score_reduction_pct"]) for c in conditions]

DATA_DIR = HERE / "figure_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
(HERE / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
for key, rows in data.items():
    with (DATA_DIR / (key + ".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 10, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.hashsalt": "LLeakM-EOS-campaign-20261102",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": .7, "savefig.dpi": 300})
blue, orange, gray = "#0077BB", "#AA4E00", "#555555"
outputs = []

def save(fig, stem):
    for suffix in ("pdf", "svg", "png"):
        target = OUT / f"{stem}.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else (
            {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
        fig.savefig(target, bbox_inches="tight", metadata=metadata, facecolor="white")
        outputs.append(target)
    plt.close(fig)

def point(ax, row, y, color, marker="o", filled=True):
    ax.errorbar(row["mean"], y,
                xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]],
                fmt=marker, color=color, markerfacecolor=color if filled else "white",
                markersize=5, capsize=3, elinewidth=1.2, zorder=3)

# Separate panels prevent fresh cohorts from being pooled into a seven-model ranking.
fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.5),
                         gridspec_kw={"height_ratios": [2.6, 1.8]})
fig.subplots_adjust(left=.27, right=.97, top=.94, bottom=.10, hspace=.63)
ax = axes[0]
names = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Phi-1.5", "Llama-3.2-3B", "Gemma-2-2B",
         "Phi-3.5-mini †", "TinyLlama-1.1B †"]
for i, row in enumerate(data["archived_scenario"]):
    point(ax, row, i, blue if i < 5 else gray, "o" if i < 5 else "s", i < 5)
ax.axhspan(4.5, 6.5, color="#F2F2F2", zorder=0)
ax.axhline(4.5, color="#999999", lw=.7, ls="--")
ax.set_yticks(range(7), names)
ax.set_ylim(6.6, -.6)
ax.set_title("(a) Archived references; EOS-bounded reconstruction", loc="left", pad=9)
ax.text(.04, .97, "† Reference-format diagnostics", transform=ax.transAxes,
        va="top", color=gray, fontsize=8)

ax = axes[1]
fresh = {r["model"]: r for r in data["fresh_scenario"]}
diagnostic = {r["model"]: r for r in data["archived_scenario"]}
for y, label in enumerate(["tinyllama_1_1b", "phi_3_5_mini"]):
    ax.plot([diagnostic[label]["mean"], fresh[label]["mean"]], [y + .11, y - .11],
            lw=.8, color="#AAAAAA", zorder=1)
    point(ax, diagnostic[label], y + .11, gray, "s", False)
    point(ax, fresh[label], y - .11, orange, "D")
    delta = next(r for r in data["paired_contrasts"] if r["model"] == label)
    ax.text(.30, y + .38,
            f"Paired Δφ = +{delta['mean_difference']:.3f} "
            f"[{delta['low']:.3f}, {delta['high']:.3f}]", fontsize=8,
            ha="center", va="center")
ax.set_yticks([0, 1], ["TinyLlama-1.1B", "Phi-3.5-mini"])
ax.set_ylim(1.60, -.35)
ax.set_title("(b) Matched reference rendering; EOS-bounded reconstruction", loc="left", pad=29)
ax.legend(handles=[Line2D([], [], marker="s", color=gray, mfc="white", ls="",
                           label="Independent-token reference rendering"),
                   Line2D([], [], marker="D", color=orange, ls="",
                           label="Verified full-sequence rendering")],
          loc="lower left", bbox_to_anchor=(0, 1.00), frameon=False, fontsize=8,
          handletextpad=.4, borderaxespad=0, ncol=1)
for ax in axes:
    ax.set_xlim(0, .65)
    ax.set_xticks([0, .1, .2, .3, .4, .5, .6])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#DDDDDD", lw=.65)
    ax.tick_params(axis="y", length=0)
axes[1].set_xlabel("Mean term-frequency cosine φ (scenario-bootstrap 95% CI)")
save(fig, "eos_model_results")

# Identical axes and separate models expose variation hidden by model averaging.
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
fig.subplots_adjust(left=.09, right=.98, top=.86, bottom=.20, wspace=.10)
styles = {"bucket_8": (blue, "o"), "pad_32": (orange, "s"),
          "batch_2": ("#006E60", "^"), "batch_4": ("#813F99", "v"),
          "rand_pad_8": (gray, "D")}
for ax, key, title in zip(axes, ["defense_qwen", "defense_llama"],
                          ["(a) Qwen2.5-1.5B", "(b) Llama-3.2-3B"]):
    for row in data[key]:
        c = row["condition"]
        color, marker = styles[c]
        x, y = row["overhead"], row["reduction"]
        ax.scatter(x, y, s=37, color=color, marker=marker, zorder=3)
        # Leader lines preserve the true x=0 positions of both batching points.
        position = {"bucket_8": (140, 94), "pad_32": (505, 35),
                    "batch_2": (165, 68), "batch_4": (160, 81),
                    "rand_pad_8": (150, 53)}[c]
        ax.annotate(c, (x, y), xytext=position, textcoords="data",
                    ha="right" if c == "pad_32" else "left", fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=.6,
                                    shrinkA=2, shrinkB=4))
    ax.set_title(title, loc="left", pad=9)
    ax.set_xlim(-25, 620)
    ax.set_ylim(25, 100)
    ax.set_xticks([0, 200, 400, 600])
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    ax.grid(color="#DDDDDD", lw=.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Mean-score reduction (%)")
fig.supxlabel("Realized protected-payload overhead (%)", y=.045, fontsize=9)
save(fig, "eos_defense_tradeoff")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {"sources": {str((AUDIT / n).relative_to(ROOT)): sha(AUDIT / n)
                        for n in SOURCE_NAMES},
            "generator_sha256": sha(Path(__file__)),
            "data_sha256": sha(HERE / "figure_data.json"),
            "csv_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(DATA_DIR.glob("*.csv"))},
            "outputs": {str(p.relative_to(ROOT)): sha(p) for p in outputs},
            "selector": "eos-through-first-inclusive/1",
            "scope": "Both model panels and all defense points use only the completed EOS campaign. Panel (b) compares matched reference renderings within that campaign. No old-selector result is plotted; no inference or bootstrap rerun."}
(HERE / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"outputs": list(manifest["outputs"]),
                  "series_rows": {k: len(v) for k, v in data.items()}}, indent=2))
