#!/usr/bin/env python3
"""Build all manuscript tables from the completed EOS-bounded campaign.

The self-contained table_structure.tex supplies layout, model display names and
literature metadata. Every empirical numeric body cell is replaced from the
current campaign's audited ledgers. This script never edits the manuscript.
"""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = HERE.parent / "validation"
TEMPLATE = HERE / "table_structure.tex"
PRIMARY = AUDIT / "main_fresh_factledger.json"
SECONDARY = AUDIT / "secondary_audit.json"
main, secondary = [json.loads(p.read_text()) for p in (PRIMARY, SECONDARY)]
assert main["all_checks_passed"] and secondary["status"] == "PASS"
models = {m["label"]: m for m in main["cohorts"]["cuda_full"]["models"]}
order = sorted(models, key=lambda k: models[k]["metrics"]["phi"]["mean"], reverse=True)
metadata_order = ["qwen_1_5b", "qwen_3b", "llama_3_2_3b", "gemma_2_2b",
                  "phi_1_5", "tinyllama_1_1b", "phi_3_5_mini"]
short = dict(zip(metadata_order, ["Qwen2.5-1.5B", "Qwen2.5-3B", "Llama-3.2-3B",
                                "Gemma-2-2B", "Phi-1.5", "TinyLlama-1.1B", "Phi-3.5-mini"]))
long = {k: m["model"] for k, m in models.items()}
long["tinyllama_1_1b"] = "TinyLlama-1.1B-Chat"
conditions = ["bucket_8", "pad_32", "batch_2", "batch_4", "rand_pad_8"]
pilot_models = ["qwen_1_5b", "llama_3_2_3b"]
cells = {(c["suite"], c["label"], c["condition"], c["attacker"]): c
         for c in secondary["cells"]}
contrasts = {(c["suite"], c["label"], c["condition"], c["attacker"]): c
             for c in secondary["paired_contrasts"]}
tables = {re.search(r"\\label\{([^}]+)\}", b)[1]: b
          for b in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", TEMPLATE.read_text(), re.S)}
assert len(tables) == 13
bindings = {}

def number(v, precision=4):
    return "--" if v is None else f"{v:.{precision}f}"

def interval(v, precision=4):
    return "[" + ", ".join(number(x, precision) for x in v) + "]"

def row(*values):
    return " & ".join(str(v) for v in values) + r" \\"

def escaped(s):
    return s.replace("_", r"\_")

def replace_body(label, rows, source, columns):
    before, remainder = tables[label].split(r"\midrule", 1)
    _, after = remainder.rsplit(r"\bottomrule", 1)
    tables[label] = before + "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule" + after
    bindings[label] = {"source": source, "columns": columns,
                       "data_rows": sum(" & " in r and not r.startswith(r"\multicolumn") for r in rows)}

def primary_source(cohort="cuda_full"):
    return str(PRIMARY.relative_to(ROOT)) + "#/cohorts/" + cohort + "/models"

def secondary_source(suite):
    return str(SECONDARY.relative_to(ROOT)) + "#cells,paired_contrasts; suite=" + suite

rows = []
for k in metadata_order:
    m = models[k]
    name = long[k] + ("-v1.0" if k == "tinyllama_1_1b" else "")
    rows.append(row(name, number(m["tokens_mean"], 1), number(m["reference_characters_mean"], 0),
                    number(m["token_bytes_mean_of_sample_means"], 2), number(m["token_limit_percent"], 1)))
replace_body("tab:length_covariates", rows, primary_source(),
             ["model", "tokens_mean", "reference_characters_mean", "token_bytes_mean_of_sample_means", "token_limit_percent"])

rows = []
for c in ["baseline"] + conditions:
    values = [cells[("map", k, c, "frozen")] for k in pilot_models]
    rows.append(row(escaped(c), *[f'{v["unique_first_inputs_truncated"]}/{v["n_prompts"]}' for v in values]))
replace_body("tab:encoder_coverage", rows, secondary_source("map"),
             ["condition", "qwen.unique_first_inputs_truncated/n_prompts", "llama.unique_first_inputs_truncated/n_prompts"])

rows = []
for k in pilot_models:
    if rows:
        rows.append(r"\midrule")
    for condition in ["real", "shuffled", "constant", "cross_prompt"]:
        key = ("trace_control", k, condition, "frozen")
        c = cells[key]
        delta = "--" if condition == "real" else f'${contrasts[key]["mean_phi_difference_vs_baseline"]:+.4f}$'
        rows.append(row(short[k] if condition == "real" else "", "true" if condition == "real" else condition.replace("_", "-"),
                        number(c["metric_means"]["phi"]), interval(c["phi_ci"]), delta))
replace_body("tab:trace_controls", rows, secondary_source("trace_control"),
             ["model", "condition", "metric_means.phi", "phi_ci", "mean_phi_difference_vs_baseline"])

rows = []
for i, k in enumerate(order):
    m = models[k]
    if i and m["diagnostic_series"] and not models[order[i-1]]["diagnostic_series"]:
        rows.extend([r"\midrule", r"\multicolumn{7}{l}{\emph{Reference-format diagnostics}} \\"])
    rows.append(row(long[k], "base" if k == "phi_1_5" else ("chat" if k == "tinyllama_1_1b" else "instruct"),
                    *[number(m["metrics"][f]["mean"]) for f in ["phi", "ed_norm", "rouge1_precision", "rougeL_precision"]],
                    number(m["LR_percent"], 2) + r"\%"))
replace_body("tab:main_results", rows, primary_source(),
             ["model", "type", "metrics.phi.mean", "metrics.ed_norm.mean", "metrics.rouge1_precision.mean", "metrics.rougeL_precision.mean", "LR_percent"])

rows = []
for k in order:
    m = models[k]
    rows.append(row(long[k], m["EOS_count"], number(m["EOS_percent"], 1), number(m["token_limit_percent"], 1),
                    number(m["metrics"]["phi"]["mean"]), number(m["EOS_only_phi_mean"]),
                    number(m["LR_percent"], 2), number(m["EOS_only_LR_percent"], 2)))
replace_body("tab:eos_completeness", rows, primary_source(),
             ["model", "EOS_count", "EOS_percent", "token_limit_percent", "metrics.phi.mean", "EOS_only_phi_mean", "LR_percent", "EOS_only_LR_percent"])

rows = []
for k in order:
    m = models[k]
    rows.append(row(long[k], number(m["metrics"]["phi"]["mean"]), interval(m["metrics"]["phi"]["ci_prompt_bootstrap"]),
                    number(m["LR_percent"], 2) + r"\%", interval(m["LR_wilson_percent"], 2) + r"\%"))
replace_body("tab:statistical_results", rows, primary_source(),
             ["model", "metrics.phi.mean", "metrics.phi.ci_prompt_bootstrap", "LR_percent", "LR_wilson_percent"])

rows = []
for k in order:
    m = models[k]
    rows.append(row(short[k] + (r"$^{\dagger}$" if m["diagnostic_series"] else ""),
                    number(m["metrics"]["phi"]["mean"]), interval(m["phi_ci_scenario_bootstrap"])))
replace_body("tab:scenario_sensitivity", rows, primary_source(),
             ["model", "metrics.phi.mean", "phi_ci_scenario_bootstrap"])

rows = []
for cohort in ["fresh_tinyllama_cuda", "fresh_phi_cuda"]:
    m = main["cohorts"][cohort]["models"][0]
    rows.append(row(short[m["label"]], number(m["metrics"]["phi"]["mean"]), interval(m["phi_ci_scenario_bootstrap"]),
                    *[number(m["metrics"][f]["mean"]) for f in ["ed_norm", "rouge1_precision", "rougeL_precision"]],
                    number(m["LR_percent"], 2) + r"\%", number(m["EOS_percent"], 1) + r"\%"))
replace_body("tab:fresh_results", rows, [primary_source(c) for c in ["fresh_tinyllama_cuda", "fresh_phi_cuda"]],
             ["model", "metrics.phi.mean", "phi_ci_scenario_bootstrap", "metrics.ed_norm.mean", "metrics.rouge1_precision.mean", "metrics.rougeL_precision.mean", "LR_percent", "EOS_percent"])

rows = []
for k in pilot_models:
    if rows:
        rows.append(r"\midrule")
    for condition in ["baseline"] + conditions:
        key = ("defense_pilot", k, condition, "frozen")
        c = cells[key]
        reduction = "--" if condition == "baseline" else number(contrasts[key]["score_reduction_pct"], 1) + r"\%"
        rows.append(row(short[k] if condition == "baseline" else "", escaped(condition),
                        number(c["metric_means"]["phi"]), interval(c["phi_ci"]), reduction,
                        number(c["observed_payload_overhead_pct_mean"], 1) + r"\%", number(c["observed_units_mean"], 0)))
replace_body("tab:defense_results", rows, secondary_source("defense_pilot"),
             ["model", "condition", "metric_means.phi", "phi_ci", "score_reduction_pct", "observed_payload_overhead_pct_mean", "observed_units_mean"])

rows = []
for k in pilot_models:
    if rows:
        rows.append(r"\midrule")
    for condition, attacker in [("baseline", "frozen")] + [(c, a) for c in conditions for a in ["frozen", "adaptive"]]:
        key = ("map", k, condition, attacker)
        c = cells[key]
        reduction = "0.0" if condition == "baseline" else number(contrasts[key]["score_reduction_pct"], 1)
        if reduction.startswith("-"):
            reduction = "$" + reduction + "$"
        rows.append(row(short[k] if condition == "baseline" else "", escaped(condition), attacker,
                        number(c["metric_means"]["phi"]), interval(c["phi_ci"]), number(c["between_seed_sd"]), reduction))
replace_body("tab:robustness", rows, secondary_source("map"),
             ["model", "condition", "attacker", "metric_means.phi", "phi_ci", "between_seed_sd", "score_reduction_pct"])

# Protocol counts are freshly derived from the campaign ledgers; descriptive
# budgets are fixed protocol settings, not empirical reconstruction outcomes.
n_main = main["cohorts"]["cuda_full"]["n"]
n_fresh = sum(main["cohorts"][c]["n"] for c in ["fresh_tinyllama_cuda", "fresh_phi_cuda"])
ns = {suite: sum(c["n"] for c in secondary["cells"] if c["suite"] == suite)
      for suite in ["trace_control", "defense_pilot", "map"]}
assert [n_main, n_fresh, ns["trace_control"], ns["defense_pilot"], ns["map"]] == [2100, 600, 240, 360, 3300]
rows = [
    row(r"Archived main; \ref{tab:main_results}--\ref{tab:scenario_sensitivity}",
        "Archived emitted text; seven models, 300 per model; two diagnostic series", n_main, 512, "All / 3", "Recorded"),
    row(r"Verified rendering; \ref{tab:fresh_results}",
        "Verified full-sequence decoding; 300 per model, TinyLlama and Phi-3.5-mini", n_fresh, 512, "All / 3", "Recorded"),
    row(r"Trace controls; \ref{tab:trace_controls}", r"Archived text; two models, $30\times4$ per model",
        ns["trace_control"], 96, r"$\leq3$ / 3", "Pilot"),
    row(r"Defense pilot; \ref{tab:defense_results}", r"Archived text; two models, $30\times6$ per model",
        ns["defense_pilot"], 96, r"$\leq3$ / 3", "Pilot"),
    row(r"MAP study; \ref{tab:robustness}", r"Archived text; two models, $30\times5\times11$ per model",
        ns["map"], 512, "First / 1", "Common, five")]
replace_body("tab:protocol_overview", rows, [str(PRIMARY.relative_to(ROOT)), str(SECONDARY.relative_to(ROOT))],
             ["experiment", "references and design", "sum n", "protocol victim limit", "requested segments / candidates", "seed rule"])

# Checkpoint names, literature parameter counts, tokenizer families and original
# victim-response directory identifiers are source metadata, not old results.
for label in ["tab:model_metadata", "tab:run_ids"]:
    body = tables[label].split(r"\midrule", 1)[1].rsplit(r"\bottomrule", 1)[0]
    bindings[label] = {"source": "current campaign source metadata; display/literature metadata in table_structure.tex",
                       "columns": "checkpoint/tokenizer/parameter metadata" if label == "tab:model_metadata" else "source-response directory identifiers",
                       "data_rows": len([r for r in body.splitlines() if " & " in r]),
                       "empirical_numeric_results": False}

# Captions describe the current EOS-bounded reconstruction and keep source
# reference formatting distinct from the selector under evaluation.
for label in tables:
    tables[label] = tables[label].replace("newly generated, validated references", "verified full-sequence references")
    tables[label] = tables[label].replace("newly generated, verified full-sequence references", "verified full-sequence references")
tables["tab:protocol_overview"] = tables["tab:protocol_overview"].replace(
    "All experiments score against the full reference", "All reconstructions use the EOS-bounded selector, score against the full reference")
# The frozen template already contains the EOS caption; do not prepend or append
# wording on regeneration. This makes the released generator idempotent.

(HERE / "tables.json").write_text(json.dumps(tables, indent=2) + "\n")
(HERE / "tables.tex").write_text("\n\n".join(tables.values()) + "\n")
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {"status": "GENERATED", "selector": "eos-through-first-inclusive/1",
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [PRIMARY, SECONDARY]},
            "template_structure_source": str(TEMPLATE.relative_to(ROOT)),
            "template_structure_sha256": sha(TEMPLATE), "generator_sha256": sha(Path(__file__)),
            "table_bindings": bindings, "table_count": len(tables),
            "data_row_count": sum(v["data_rows"] for v in bindings.values()),
            "output_sha256": {p.name: sha(p) for p in [HERE / "tables.json", HERE / "tables.tex"]},
            "scope": "Every empirical result is drawn from the new EOS campaign; metadata and reference identifiers are descriptive source information. MAP values are recomputed campaign cells, even where unchanged."}
(HERE / "table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: manifest[k] for k in ["table_count", "data_row_count"]}))
