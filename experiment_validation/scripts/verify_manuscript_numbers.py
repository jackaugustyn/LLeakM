"""Compare rounded values in article1/template.tex against analysis artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "article1" / "template.tex"
ANALYSIS = ROOT / "experiment_validation" / "analysis"


def rnd(x: float, n: int) -> float:
    return float(f"{x:.{n}f}")


def main() -> int:
    tex = TEX.read_text(encoding="utf-8")
    stats = json.loads((ANALYSIS / "model_stats.json").read_text())
    pairs = json.loads((ANALYSIS / "pairwise_tests.json").read_text())
    defenses = json.loads((ANALYSIS / "defense_eval.json").read_text())
    mismatches: list[str] = []

    # Main table uses 4-decimal phi/ed/r1/rl and 3-decimal LR percent.
    for m in stats:
        phi = f"{m['phi_mean']:.4f}"
        ed = f"{m['ed_norm_mean']:.4f}"
        r1 = f"{m['r1_mean']:.4f}"
        rl = f"{m['rl_mean']:.4f}"
        lr = f"{m['asr']:.3f}\\%"
        # Some tables use 2-decimal LR without trailing zeros, e.g. 8.67\%
        lr2 = f"{m['asr']:.2f}\\%"
        if phi not in tex:
            mismatches.append(f"missing phi {m['model']}={phi}")
        if ed not in tex:
            mismatches.append(f"missing ed {m['model']}={ed}")
        if r1 not in tex:
            mismatches.append(f"missing r1 {m['model']}={r1}")
        if rl not in tex:
            mismatches.append(f"missing rl {m['model']}={rl}")
        if lr not in tex and lr2 not in tex:
            mismatches.append(f"missing LR {m['model']}={lr} or {lr2}")
        lo, hi = m["phi_ci"]
        ci = f"[{lo:.4f}, {hi:.4f}]"
        if ci not in tex:
            mismatches.append(f"missing phi CI {m['model']}={ci}")

    n_sig = sum(1 for p in pairs if p["significant_0_05"])
    if f"{n_sig} of {len(pairs)}" not in tex and f"{n_sig} of 21" not in tex:
        mismatches.append(f"missing Holm count {n_sig} of {len(pairs)}")

    # Named pairwise exceptions from the manuscript are checked against JSON:
    # the two smallest |Holm| non-significant or borderline pairs plus the
    # largest mean difference, formatted as in template.tex.
    ordered = sorted(pairs, key=lambda p: p["p_holm"], reverse=True)
    ns = [p for p in pairs if not p["significant_0_05"]]
    biggest = max(pairs, key=lambda p: abs(p["mean_diff_phi"]))
    for p in ns + [biggest]:
        diff = f"{p['mean_diff_phi']:+.4f}".lstrip("+")
        # manuscript may use unsigned for positive diffs
        if f"{p['mean_diff_phi']:.4f}" not in tex and f"{p['mean_diff_phi']:+.4f}" not in tex:
            mismatches.append(f"pairwise diff {p['a']}/{p['b']}={p['mean_diff_phi']:.4f}")
        r = f"{p['rank_biserial']:.3f}"
        if r not in tex and f"{p['rank_biserial']:+.3f}" not in tex:
            mismatches.append(f"pairwise r {p['a']}/{p['b']}={r}")
        if p["p_holm"] >= 1e-4:
            ph = f"{p['p_holm']:.4f}"
            if ph not in tex:
                mismatches.append(f"pairwise p_holm {p['a']}/{p['b']}={ph}")

    for d in defenses:
        if d["defense"] == "baseline":
            continue
        red = f"{d['leakage_reduction_pct']:.1f}\\%"
        phi = f"{d['phi_mean']:.4f}"
        if red not in tex:
            mismatches.append(f"missing defense reduction {d['model']} {d['defense']}={red}")
        if phi not in tex:
            mismatches.append(f"missing defense phi {d['model']} {d['defense']}={phi}")

    robustness_path = ANALYSIS / "robustness_eval.json"
    if robustness_path.exists():
        robustness = json.loads(robustness_path.read_text())
        if robustness.get("complete"):
            if "22" not in tex:
                mismatches.append("missing robustness cell count 22")
            for row in robustness.get("cells", []):
                phi = f"{row['phi_mean']:.4f}"
                if phi not in tex:
                    mismatches.append(f"missing robustness phi {row['label']} {row['defense']} {row['attacker']}={phi}")
                red = f"{row['score_reduction_pct']:.1f}"
                if red not in tex and f"{abs(row['score_reduction_pct']):.1f}" not in tex:
                    mismatches.append(
                        f"missing robustness reduction {row['label']} {row['defense']} {row['attacker']}={red}"
                    )

    topic = (ANALYSIS / "tables_topic.tex").read_text(encoding="utf-8")
    for val in re.findall(r"\d\.\d{4}", topic):
        if val not in tex:
            mismatches.append(f"missing topic value {val}")

    out = ANALYSIS / "manuscript_number_verification.json"
    payload = {"ok": not mismatches, "mismatches": mismatches, "n_checked_models": len(stats)}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if mismatches:
        print(f"[verify] {len(mismatches)} mismatches")
        for line in mismatches:
            print("  -", line)
        return 1
    print("[verify] manuscript numbers match released analysis artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
