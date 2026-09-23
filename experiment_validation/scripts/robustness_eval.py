"""Repeated-seed and defense-aware reconstruction evaluation.

This experiment addresses three threats to validity in the original campaign:

* the stochastic reconstructor is repeated for several seeds;
* the same decoder seed is used for every condition of a prompt;
* a defense-aware attacker is calibrated on prompt-disjoint traces.

The adaptive attacker does not see test responses.  It learns the empirical
inverse of each defense (clean token length conditioned on the defended
observation) from the remaining prompts of the same victim model, then passes
the inferred clean-length trace to the unchanged Weiss-style T5 reconstructor.
For batching, it learns likely token-length tuples conditioned on aggregate
length and falls back to a MAP dynamic program.  This is a calibrated adaptive
attacker, not end-to-end T5 retraining, and should be described as such.

By default all heuristic response segments are reconstructed.  The script is
resumable and writes a completeness flag; partial smoke-test output is never
rendered as a publication table.
"""

from __future__ import annotations

import argparse
import hashlib
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
os.environ.setdefault("LLEAKM_RECONSTRUCT_DEVICE", "cpu")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weiss_reconstruction import reconstruct

RESULTS = ROOT / "experiment_validation" / "results"
LOGDIR = ROOT / "logs"
OUT = ROOT / "experiment_validation" / "analysis"
WORD_RE = re.compile(r"[A-Za-z0-9']+")

RUNS = {
    "qwen_1_5b": ("Qwen2.5-1.5B-Instruct", "run_full_20260920_qwen_1_5b_full"),
    "qwen_3b": ("Qwen2.5-3B-Instruct", "run_full_20260920_qwen_3b_full"),
    "phi_1_5": ("Phi-1.5", "run_full_20260920_phi_1_5_full"),
    "llama_3_2_3b": ("Llama-3.2-3B-Instruct", "run_full_20260920_llama_3_2_3b_full"),
    "gemma_2_2b": ("Gemma-2-2B-it", "run_full_20260920_gemma_2_2b_full"),
    "tinyllama_1_1b": ("TinyLlama-1.1B-Chat", "run_full_20260920_tinyllama_1_1b_full"),
    "phi_3_5_mini": ("Phi-3.5-mini-instruct", "run_full_20260920_phi_3_5_mini_full"),
}
DEFAULT_LABELS = ["qwen_1_5b", "llama_3_2_3b"]
DEFAULT_CONDITIONS = ["baseline", "bucket_8", "pad_32", "batch_2", "batch_4", "rand_pad_8"]
DEFAULT_SEEDS = [20260706, 20260707, 20260708, 20260709, 20260710]


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big") & 0x7FFFFFFF


def common_decoder_seed(seed_index: int, prompt_idx: int) -> int:
    """Condition-independent seed used for one prompt/replicate pair."""
    return stable_seed("decoder", seed_index, prompt_idx)


def tf_cosine(a: str, b: str) -> float:
    ta, tb = WORD_RE.findall(a.lower()), WORD_RE.findall(b.lower())
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    vocab = set(ca) | set(cb)
    dot = float(sum(ca[w] * cb[w] for w in vocab))
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


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
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1] / max(1, len(ref))


def rouge1_precision(ref: str, pred: str) -> float:
    r, p = WORD_RE.findall(ref.lower()), WORD_RE.findall(pred.lower())
    if not r or not p:
        return 0.0
    return sum((Counter(r) & Counter(p)).values()) / len(p)


def load_samples(run_dir: str) -> list[dict]:
    rows: list[dict] = []
    with (RESULTS / run_dir / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append({
                "idx": int(row["idx"]),
                "topic": row["topic"],
                "run_id": row["run_id"],
                "response_text": row.get("response_text", ""),
                "victim_model_id": row.get("victim_model_id", ""),
                "response_complete": bool(row.get("response_complete", False)),
            })
    return rows


def token_lengths(run_id: str) -> list[int] | None:
    path = LOGDIR / f"{run_id}.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [int(step["token_utf8_len"]) for step in raw.get("steps", [])]


def split_rows(rows: list[dict], test_per_topic: int, complete_only: bool = False) -> tuple[list[dict], list[dict]]:
    """Deterministic topic-stratified test/calibration split."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if complete_only and not row["response_complete"]:
            continue
        grouped[row["topic"]].append(row)
    test, calibration = [], []
    for topic in sorted(grouped):
        group = sorted(grouped[topic], key=lambda row: row["idx"])
        test.extend(group[:test_per_topic])
        calibration.extend(group[test_per_topic:])
    return sorted(test, key=lambda row: row["idx"]), sorted(calibration, key=lambda row: row["idx"])


def transform_trace(name: str, clean: list[int], rng: np.random.Generator) -> list[int]:
    if name == "baseline":
        return list(clean)
    if name == "bucket_8":
        return [int(math.ceil(x / 8.0) * 8) for x in clean]
    if name == "pad_32":
        return [32] * len(clean)
    if name == "batch_2":
        return [sum(clean[i:i + 2]) for i in range(0, len(clean), 2)]
    if name == "batch_4":
        return [sum(clean[i:i + 4]) for i in range(0, len(clean), 4)]
    if name == "rand_pad_8":
        return [int(x + rng.integers(0, 9)) for x in clean]
    raise ValueError(f"unknown condition: {name}")


class AdaptiveLengthDecoder:
    """Empirical MAP inverse learned from prompt-disjoint calibration traces."""

    def __init__(self, defense: str, calibration_draws: int = 16):
        self.defense = defense
        self.calibration_draws = calibration_draws
        self.token_prior: Counter[int] = Counter()
        self.inverse: dict[int, Counter[int]] = defaultdict(Counter)
        self.batch_inverse: dict[tuple[int, int], Counter[tuple[int, ...]]] = defaultdict(Counter)
        self.tail_sizes: Counter[int] = Counter()

    @property
    def batch_size(self) -> int | None:
        if self.defense == "batch_2":
            return 2
        if self.defense == "batch_4":
            return 4
        return None

    def fit(self, sequences: list[list[int]], seed: int) -> None:
        for sequence_index, clean in enumerate(sequences):
            self.token_prior.update(clean)
            if self.batch_size:
                k = self.batch_size
                for pos in range(0, len(clean), k):
                    part = tuple(clean[pos:pos + k])
                    self.batch_inverse[(sum(part), len(part))][part] += 1
                    if pos + k >= len(clean):
                        self.tail_sizes[len(part)] += 1
                continue

            draws = self.calibration_draws if self.defense == "rand_pad_8" else 1
            for draw in range(draws):
                rng = np.random.default_rng(
                    stable_seed("calibration", seed, self.defense, sequence_index, draw)
                )
                observed = transform_trace(self.defense, clean, rng)
                for z, x in zip(observed, clean):
                    self.inverse[z][x] += 1

    def _mode(self, counts: Counter[int]) -> int:
        if counts:
            return min(counts, key=lambda value: (-counts[value], value))
        if self.token_prior:
            return min(self.token_prior, key=lambda value: (-self.token_prior[value], value))
        return 1

    def _map_tuple(self, total: int, size: int) -> tuple[int, ...]:
        learned = self.batch_inverse.get((total, size))
        if learned:
            return min(learned, key=lambda value: (-learned[value], value))

        support = sorted(self.token_prior) or list(range(0, 18))
        denom = sum(self.token_prior.values()) + len(support)
        logp = {x: math.log((self.token_prior[x] + 1) / denom) for x in support}
        states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
        for _ in range(size):
            nxt: dict[int, tuple[float, tuple[int, ...]]] = {}
            for subtotal, (score, values) in states.items():
                for value in support:
                    new_total = subtotal + value
                    if new_total > total:
                        continue
                    candidate = (score + logp[value], values + (value,))
                    old = nxt.get(new_total)
                    if old is None or candidate[0] > old[0] or (
                        candidate[0] == old[0] and candidate[1] < old[1]
                    ):
                        nxt[new_total] = candidate
            states = nxt
        if total in states:
            return states[total][1]
        fallback = self._mode(self.token_prior)
        return tuple([fallback] * size)

    def decode(self, observed: list[int]) -> list[int]:
        if self.defense == "baseline":
            return list(observed)
        if self.batch_size:
            result: list[int] = []
            k = self.batch_size
            for pos, total in enumerate(observed):
                if pos < len(observed) - 1:
                    size = k
                else:
                    candidates = []
                    for size_candidate in range(1, k + 1):
                        learned = self.batch_inverse.get((total, size_candidate))
                        count = sum(learned.values()) if learned else 0
                        candidates.append((count * max(1, self.tail_sizes[size_candidate]), size_candidate))
                    size = max(candidates)[1]
                result.extend(self._map_tuple(total, size))
            return result
        return [self._mode(self.inverse.get(value, Counter())) for value in observed]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in parse_csv(value)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("decoder seeds must be unique")
    return seeds


def protocol_id(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def completed_keys(path: Path, current_protocol: str) -> set[tuple]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("protocol_id") != current_protocol:
                continue
            done.add((row["label"], int(row["idx"]), row["defense"], row["attacker"], int(row["decoder_seed"])))
    return done


def crossed_bootstrap_ci(matrix: np.ndarray, seed: int, n_boot: int) -> list[float]:
    """Bootstrap prompt and decoder-seed axes independently."""
    rng = np.random.default_rng(seed)
    n_prompt, n_seed = matrix.shape
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pi = rng.integers(0, n_prompt, size=n_prompt)
        si = rng.integers(0, n_seed, size=n_seed)
        means[b] = matrix[np.ix_(pi, si)].mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def aggregate(sample_path: Path, config: dict, current_protocol: str, n_boot: int) -> dict:
    rows = []
    with sample_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("protocol_id") == current_protocol:
                rows.append(row)

    by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_cell[(row["label"], row["defense"], row["attacker"])].append(row)

    summaries = []
    matrices: dict[tuple[str, str, str], tuple[list[int], list[int], np.ndarray]] = {}
    expected_per_cell = config["expected_prompts_per_model"] * len(config["decoder_seeds"])
    for key in sorted(by_cell):
        cell_rows = by_cell[key]
        idxs = sorted({int(row["idx"]) for row in cell_rows})
        seeds = sorted({int(row["seed_index"]) for row in cell_rows})
        value_map = {(int(row["idx"]), int(row["seed_index"])): float(row["phi"]) for row in cell_rows}
        cell_complete = len(value_map) == expected_per_cell and len(idxs) == config["expected_prompts_per_model"]
        matrix = np.array([[value_map.get((idx, seed), np.nan) for seed in seeds] for idx in idxs])
        finite = matrix[np.isfinite(matrix)]
        seed_means = [float(np.nanmean(matrix[:, col])) for col in range(matrix.shape[1])]
        ci = (
            crossed_bootstrap_ci(matrix, stable_seed("bootstrap", *key), n_boot)
            if cell_complete and not np.isnan(matrix).any()
            else [float("nan"), float("nan")]
        )
        summaries.append({
            "label": key[0], "model": config["models"][key[0]], "defense": key[1], "attacker": key[2],
            "n_prompts": len(idxs), "n_seeds": len(seeds), "n_reconstructions": len(value_map),
            "complete": cell_complete, "phi_mean": float(finite.mean()) if len(finite) else float("nan"),
            "phi_ci_crossed": ci,
            "between_seed_sd": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            "lr_at_0_5_pct": 100.0 * float(np.mean(finite > 0.5)) if len(finite) else float("nan"),
        })
        matrices[key] = (idxs, seeds, matrix)

    baseline_means = {
        summary["label"]: summary["phi_mean"]
        for summary in summaries
        if summary["defense"] == "baseline" and summary["attacker"] == "frozen"
    }
    for summary in summaries:
        baseline = baseline_means.get(summary["label"], float("nan"))
        summary["score_reduction_pct"] = (
            100.0 * (1.0 - summary["phi_mean"] / baseline) if baseline > 0 else float("nan")
        )

    contrasts = []
    for label in config["run_labels"]:
        base_key = (label, "baseline", "frozen")
        if base_key not in matrices:
            continue
        base_idxs, base_seeds, base_matrix = matrices[base_key]
        for key, (idxs, seeds, matrix) in matrices.items():
            if key[0] != label or key == base_key or idxs != base_idxs or seeds != base_seeds:
                continue
            if np.isnan(base_matrix).any() or np.isnan(matrix).any():
                continue
            diff = matrix - base_matrix
            contrasts.append({
                "label": label, "defense": key[1], "attacker": key[2],
                "mean_phi_difference_vs_baseline": float(diff.mean()),
                "ci_crossed": crossed_bootstrap_ci(diff, stable_seed("contrast", *key), n_boot),
            })

    cells_per_model = sum(
        1 if condition == "baseline" else len(config["attackers"])
        for condition in config["conditions"]
    )
    expected_cells = len(config["run_labels"]) * cells_per_model
    complete = len(summaries) == expected_cells and all(item["complete"] for item in summaries)
    return {
        "protocol_id": current_protocol,
        "complete": complete,
        "expected_cells": expected_cells,
        "observed_cells": len(summaries),
        "config": config,
        "cells": summaries,
        "paired_contrasts": contrasts,
    }


def write_table(result: dict, path: Path) -> None:
    if not result["complete"]:
        return
    lines = ["% Auto-generated by robustness_eval.py; complete crossed seed/prompt design"]
    for row in result["cells"]:
        ci = row["phi_ci_crossed"]
        defense = row["defense"].replace("_", "\\_")
        lines.append(
            f"{row['model']} & {defense} & {row['attacker']} & "
            f"{row['n_prompts']} & {row['n_seeds']} & {row['phi_mean']:.4f} & "
            f"{{[{ci[0]:.4f}, {ci[1]:.4f}]}} & {row['between_seed_sd']:.4f} & "
            f"{row['score_reduction_pct']:.1f} \\\\"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument(
        "--run-spec", action="append", default=[], metavar="LABEL=RUN_DIRECTORY",
        help="evaluate an arbitrary run (repeatable); overrides --run-labels",
    )
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--attackers", default="frozen,adaptive", help="frozen, adaptive, or both")
    parser.add_argument("--decoder-seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--test-per-topic", type=int, default=2)
    parser.add_argument(
        "--complete-only", action="store_true",
        help="use only responses explicitly marked as EOS-complete (for long-response runs)",
    )
    parser.add_argument("--max-test-prompts", type=int, default=0, help="smoke-test cap; 0 uses all selected prompts")
    parser.add_argument("--samples-per-segment", type=int, default=3)
    parser.add_argument("--num-first-candidates", type=int, default=3)
    parser.add_argument("--max-sentences", type=int, default=0, help="0 reconstructs all available segments")
    parser.add_argument("--calibration-draws", type=int, default=16)
    parser.add_argument(
        "--min-calibration-prompts", type=int, default=200,
        help="minimum prompt-disjoint traces per model when adaptive is requested",
    )
    parser.add_argument("--trace-seed", type=int, default=20260920)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=OUT / "robustness_samples.jsonl")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    run_catalog = dict(RUNS)
    if args.run_spec:
        labels = []
        for spec in args.run_spec:
            if "=" not in spec:
                raise SystemExit(f"invalid --run-spec {spec!r}; expected LABEL=RUN_DIRECTORY")
            label, run_dir = spec.split("=", 1)
            label, run_dir = label.strip(), run_dir.strip()
            if not label or not run_dir:
                raise SystemExit(f"invalid --run-spec {spec!r}; expected LABEL=RUN_DIRECTORY")
            run_catalog[label] = (label, run_dir)
            labels.append(label)
    else:
        labels = parse_csv(args.run_labels)
    conditions = parse_csv(args.conditions)
    attackers = parse_csv(args.attackers)
    seeds = parse_seeds(args.decoder_seeds)
    unknown_labels = sorted(set(labels) - set(run_catalog))
    unknown_conditions = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unknown_labels or unknown_conditions:
        raise SystemExit(f"unknown labels={unknown_labels}; unknown conditions={unknown_conditions}")
    if not attackers or set(attackers) - {"frozen", "adaptive"}:
        raise SystemExit("attackers must be frozen, adaptive, or frozen,adaptive")
    if "baseline" not in conditions:
        raise SystemExit("conditions must include baseline for paired score-reduction estimates")
    if args.test_per_topic < 1 or not seeds:
        raise SystemExit("test-per-topic and decoder-seeds must be non-empty/positive")

    selected: dict[str, tuple[list[dict], list[dict]]] = {}
    model_names: dict[str, str] = {}
    for label in labels:
        configured_name, run_dir = run_catalog[label]
        test, calibration = split_rows(
            load_samples(run_dir), args.test_per_topic, complete_only=args.complete_only
        )
        if args.complete_only:
            topic_counts = Counter(row["topic"] for row in test)
            short = sorted(topic for topic in {row["topic"] for row in load_samples(run_dir)}
                           if topic_counts[topic] < args.test_per_topic)
            if short:
                raise SystemExit(
                    f"{label} lacks {args.test_per_topic} EOS-complete test responses in topics: {short}"
                )
        all_rows = test + calibration
        model_names[label] = next(
            (row["victim_model_id"] for row in all_rows if row["victim_model_id"]),
            configured_name,
        )
        test = [row for row in test if token_lengths(row["run_id"])]
        calibration = [row for row in calibration if token_lengths(row["run_id"])]
        if args.max_test_prompts:
            test = test[:args.max_test_prompts]
        selected[label] = (test, calibration)

    if "adaptive" in attackers:
        insufficient = {
            label: len(parts[1]) for label, parts in selected.items()
            if len(parts[1]) < args.min_calibration_prompts
        }
        if insufficient:
            raise SystemExit(
                "insufficient prompt-disjoint calibration traces for adaptive attack: "
                f"{insufficient}; collect/restore the missing logs or use --attackers frozen"
            )

    prompt_counts = {label: len(parts[0]) for label, parts in selected.items()}
    if len(set(prompt_counts.values())) != 1:
        raise SystemExit(f"unbalanced test prompt counts: {prompt_counts}")
    expected_prompts = next(iter(prompt_counts.values()), 0)
    config = {
        "schema_version": 1,
        "run_labels": labels,
        "run_directories": {label: run_catalog[label][1] for label in labels},
        "models": model_names,
        "conditions": conditions,
        "attackers": attackers,
        "decoder_seeds": seeds,
        "common_random_numbers": True,
        "decoder_seed_derivation": "blake2s('decoder', seed_index, prompt_idx)",
        "trace_seed": args.trace_seed,
        "test_per_topic": args.test_per_topic,
        "complete_only": args.complete_only,
        "max_test_prompts": args.max_test_prompts,
        "expected_prompts_per_model": expected_prompts,
        "calibration_prompt_counts": {label: len(parts[1]) for label, parts in selected.items()},
        "calibration_draws_rand_pad_8": args.calibration_draws,
        "minimum_calibration_prompts": args.min_calibration_prompts,
        "samples_per_segment": args.samples_per_segment,
        "num_first_candidates": args.num_first_candidates,
        "max_sentences": args.max_sentences,
        "all_segments": args.max_sentences <= 0,
        "adaptive_attacker": "prompt-disjoint empirical MAP inverse plus unchanged T5 decoder",
    }
    current_protocol = protocol_id(config)
    n_modes = sum(1 if condition == "baseline" else len(attackers) for condition in conditions)
    expected_runs = len(labels) * expected_prompts * len(seeds) * n_modes
    print(f"[robustness] protocol={current_protocol} prompts={prompt_counts} decoder calls={expected_runs}", flush=True)
    if args.plan:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.output.with_name(args.output.stem + "_config.json")
    if args.overwrite and args.output.exists():
        args.output.unlink()
    elif args.output.exists() and config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("protocol_id") != current_protocol:
            raise SystemExit(
                f"{args.output} belongs to protocol {previous.get('protocol_id')}; "
                "choose another --output or pass --overwrite"
            )
    config_path.write_text(
        json.dumps({"protocol_id": current_protocol, **config}, indent=2), encoding="utf-8"
    )

    if not args.aggregate_only:
        completed = completed_keys(args.output, current_protocol)
        mode = "a" if args.output.exists() and not args.overwrite else "w"
        with args.output.open(mode, encoding="utf-8") as output:
            started = time.time()
            new_count = 0
            for label in labels:
                pretty = model_names[label]
                test_rows, calibration_rows = selected[label]
                calibration_sequences = [token_lengths(row["run_id"]) for row in calibration_rows]
                calibration_sequences = [seq for seq in calibration_sequences if seq]
                adaptive = {}
                for condition in conditions:
                    decoder = AdaptiveLengthDecoder(condition, args.calibration_draws)
                    decoder.fit(calibration_sequences, seed=stable_seed(args.trace_seed, label, condition))
                    adaptive[condition] = decoder

                for row in test_rows:
                    clean = token_lengths(row["run_id"])
                    if not clean:
                        continue
                    for condition in conditions:
                        trace_seed = stable_seed("trace", args.trace_seed, label, row["idx"], condition)
                        observed = transform_trace(condition, clean, np.random.default_rng(trace_seed))
                        attack_inputs = []
                        if condition == "baseline" or "frozen" in attackers:
                            attack_inputs.append(("frozen", observed))
                        if condition != "baseline" and "adaptive" in attackers:
                            attack_inputs.append(("adaptive", adaptive[condition].decode(observed)))
                        for attacker, attack_input in attack_inputs:
                            for seed_index in seeds:
                                decode_seed = common_decoder_seed(seed_index, row["idx"])
                                key = (label, row["idx"], condition, attacker, decode_seed)
                                if key in completed:
                                    continue
                                rec = reconstruct(
                                    attack_input,
                                    num_first_candidates=args.num_first_candidates,
                                    max_sentences=args.max_sentences,
                                    samples_per_segment=args.samples_per_segment,
                                    seed=decode_seed,
                                )
                                pred = rec.full_text
                                record = {
                                    "protocol_id": current_protocol,
                                    "model": pretty, "label": label, "idx": row["idx"], "topic": row["topic"],
                                    "run_id": row["run_id"], "defense": condition, "attacker": attacker,
                                    "source_response_complete": row["response_complete"],
                                    "seed_index": seed_index, "decoder_seed": decode_seed, "trace_seed": trace_seed,
                                    "n_clean_tokens": len(clean), "n_observed_units": len(observed),
                                    "n_attacker_input_units": len(attack_input),
                                    "reconstructed_segments": rec.sentence_count,
                                    "available_segments": rec.available_segment_count,
                                    "phi": tf_cosine(row["response_text"], pred),
                                    "ed_norm": levenshtein_norm(row["response_text"], pred),
                                    "r1_precision": rouge1_precision(row["response_text"], pred),
                                    "pred": pred,
                                }
                                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                                output.flush()
                                completed.add(key)
                                new_count += 1
                    print(
                        f"[robustness] {label} idx={row['idx']} new={new_count} "
                        f"elapsed={time.time() - started:.0f}s",
                        flush=True,
                    )

    result = aggregate(args.output, config, current_protocol, args.bootstrap_resamples)
    result_path = args.output.with_name(args.output.stem.replace("_samples", "_eval") + ".json")
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    table_path = args.output.with_name(args.output.stem.replace("_samples", "_table") + ".tex")
    write_table(result, table_path)
    print(
        f"[robustness] complete={result['complete']} cells={result['observed_cells']}/"
        f"{result['expected_cells']} -> {result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
