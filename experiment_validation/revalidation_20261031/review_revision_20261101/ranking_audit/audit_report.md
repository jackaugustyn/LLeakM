# Bounded replay of the candidate selector — R1-W1

**Completed 1 November 2026.** All 24 saved first-segment selections were reproduced exactly on the original CUDA/PyTorch/Transformers stack. Re-ranking the same generated candidates through the first EOS (including EOS) selected a different text in **18 of the 24 deliberately selected cases**. This is direct evidence that the mechanism can affect actual reconstruction inputs; the count is **not an estimate of its prevalence in the complete campaign**.

## Design and results

The input inventory was saved before inference. For each of seven archived cohorts and two fresh-reference cohorts, the audit selected the minimum and maximum full first-encoder input length, resolving ties deterministically by prompt index (18 cases). Six defense-pilot cases cover baseline, bucket_8, pad_32, batch_2, batch_4 and rand_pad_8; models alternate Qwen/Llama, with the lowest prompt index in each condition. No candidate outcome entered selection. Trace-control conditions and middle segments were not sampled.

| Source | Audited first segments | Changed selected text |
|---|---:|---:|
| Seven archived-reference cohorts | 14 | 11 |
| Two fresh-reference cohorts | 4 | 3 |
| Defense pilot | 6 | 4 |
| Total | 24 | 18 |

The 24 calls produced **72 candidates**. **38 candidates** have forced PAD positions after their first EOS, and all 38 of these tails contain at least one nonfinite transition score. No candidate has a nonfinite score before or at its first EOS. The historical full-row sum can therefore reduce a completed candidate's confidence to negative infinity because of positions outside the generated response. Scores in this audit are normalized **processed sampling distributions** (including top-k and no-repeat-ngram transformations), not unprocessed model sequence likelihoods.

The audit called the original `weiss_reconstruction._sample_and_rank` unchanged, intercepting its generated sequences and transition scores. Input token IDs matched saved inputs in every case. Decoder seeds were set after loading the model, exactly as in the campaign. The first-sentence checkpoint revision, weight/config/tokenizer hashes, FP32 precision, generation parameters, PyTorch 2.5.1+cu124 and Transformers 4.46.3 match the original protocols. The original selected text matched `first_prediction` in **24/24** records. Generation plus model load took **28.7 seconds** after checkpoint-file validation.

`results.jsonl` retains every candidate's full output IDs, normalized selected-token transition scores, first EOS offset, prefix/tail counts, both scores, both rankings and both selected texts. Full-vocabulary logits are not retained. Nonfinite values use explicit strings rather than fabricated numeric sentinels.

## Meaning for the article

The existing manuscript tables remain valid observations of the **frozen, PAD-inclusive selector** and must be attributed to that exact implementation. They do not estimate the performance of an EOS-aware selector. The bounded audit gives no full-response effect estimate and cannot determine whether any aggregate result, model ordering or Holm decision would change.

If the article were instead to report an EOS-aware reconstruction pipeline, simply re-scoring the saved final predictions would be insufficient. Choosing a different first or middle segment changes the context supplied to the next segment. The complete affected multi-candidate protocols (archived main, fresh-reference, controls and defense pilot) would need separate, versioned end-to-end replays and new aggregation. The **3300 one-candidate MAP records do not perform candidate ranking**, so this defect does not require their repetition. No such full-campaign rerun is represented by this audit.

This implementation follows the review's minimal acceptance path: disclose and retain the literal frozen pipeline, with additional bounded empirical evidence. The original reconstructor, historical predictions and published-table inputs were not edited. `eos_scoring.py` is an **audit-only** alternative, not a silently substituted production implementation.

## Validation and reproduction

- `select_inputs.py` reconstructs all 24 inventory entries from original source records, including source-record hashes.
- `test_eos_scoring.py`: six tests pass, including the reviewer's counterexample through the real Transformers score function, multiple EOS IDs, PAD=EOS, no EOS, PAD before EOS, nonfinite prefixes and invalid alignment.
- `verify_audit.py` independently checks all saved token IDs/EOS positions, prefix and full sums, rankings, selection counts and source-input identities. GPU/CPU reduction differences use absolute tolerance 2e-5 and relative tolerance 2e-6; no score is altered to meet a check.
- `attempt_01/` retains an incomplete two-record instrumentation attempt. It stopped on an exact CPU-vs-GPU floating-point comparison, before the final run used identical per-row GPU reductions. It is excluded from the 24-case denominator.

From the repository root, use system `python3`:

```bash
python3 experiment_validation/revalidation_20261031/review_revision_20261101/ranking_audit/select_inputs.py
python3 -m unittest discover -s experiment_validation/revalidation_20261031/review_revision_20261101/ranking_audit -p 'test_eos_scoring.py' -v
python3 experiment_validation/revalidation_20261031/review_revision_20261101/ranking_audit/verify_audit.py
```

`run_audit.py` documents the inference procedure and refuses to overwrite an existing `results.jsonl`. Copy the audit source files and `inputs.jsonl` into a separate directory at the same repository depth to make a new run; retain the original result directory as immutable evidence. Reproduction requires the pinned model files and original CUDA environment. The incomplete `.venv-analysis` directory is not the execution environment.
