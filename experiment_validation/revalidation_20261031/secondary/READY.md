## Material Passport

- Origin Skill: ARS-Codex / experiment-agent
- Origin Mode: run preparation and reproducibility validation
- Origin Date: 2026-10-31
- Verification Status: READY — inventory and integrity checks passed; new secondary T5 inference has not been executed.
- Version Label: secondary_rerun_v1

Prepared `rerun_secondary.py` for a separate **3900-record** campaign. Run after the primary GPU job exits:

```bash
python3 experiment_validation/revalidation_20261031/secondary/rerun_secondary.py \
  --output experiment_validation/revalidation_20261031/secondary/cuda_full --device cuda
```

Then independently validate stored inputs, texts, metrics and encoder tokenization, regenerate statistics, and require a complete design. This command does not load model weights or require GPU access (the `cuda` value identifies the stored protocol):

```bash
python3 experiment_validation/revalidation_20261031/secondary/rerun_secondary.py \
  --output experiment_validation/revalidation_20261031/secondary/cuda_full --device cuda \
  --aggregate-only --require-complete
```

`--require-complete` returns exit code 2 for a partial campaign. A successful complete run writes `status.json` with `complete: true`, `expected: 3900`, `completed: 3900`, and `summary.json` with `complete: true`, `expected: 3900`, `n: 3900`. All three suite-completeness flags and all 42 cell flags must be true. The supervisor should check these files before advancing.

### Scope and provenance

- **240 controls**: two models × 30 archived 96-token responses × real/shuffled/constant/cross-prompt; up to three segments and three candidates per segment. The cyclic donor rule is explicit: `(target_idx % 30) + 1`, matching actual historical donors for the selected indices 1–30.
- **360 defense records**: the same 60 responses × six conditions, with the same segment/candidate budgets. Pilots use new stable BLAKE2s seeds, not inferred historical seeds. Random padding's seed, full observed vector and realized payload overhead are retained.
- **3300 MAP records**: 512-token source traces, one segment and one candidate; archived 30-test/270-calibration split per model, 16 random-padding calibration draws, five decoder seeds and the same transformation/inverse algorithms. The actual plan matches all 3300 archived keys, decoder seeds, transformation seeds and input-unit counts. This is not a claim that unavailable historical encoded inputs or T5 outputs were recovered exactly.
- Both T5 revisions are pinned and loaded offline in float32. The protocol hashes the two weights plus model, tokenizer, special-token and generation configuration files, all relevant code, source files, calibration inventory and execution-library versions.
- `input_inventory.jsonl` preserves reference text, clean lengths, transformed lengths, inverse inputs, seeds, costs and source hashes. `calibration.json` preserves all 540 calibration records. Existing inventories/protocols must match byte-for-byte; output locking prevents concurrent appenders.
- Every result preserves full prediction, selected output of every T5 call, input string, full token IDs and retained token IDs. Validation recomputes exact segmentation, serialization, preceding context, pinned tokenization, text concatenation, ED, all ROUGE measures and lexical cosine. Short segment runs are accepted only when the original reconstructor stopped after an empty middle output; counts and empty-first outputs remain visible.
- Full-design statistics include all metric means, prompt bootstrap for pilots, crossed prompt/seed bootstrap for MAP, paired differences, pilot signed-rank tests and separate Holm families, realized cost, frame counts and segment shortfalls. Partial cells have descriptive means only; contrasts require the complete relevant suite.
- Original emitted-text references remain unchanged. This campaign does not perform new victim generation or repair historical detokenization, and differing encoder coverage between transformations remains part of the protocol.

### Checks and readiness

Final script SHA-256: `3508b2a82e948de53f0355f7cd64433a6d494ad929b224a9747951f1cc117621`.

- Final `--plan`: **3900 unique tasks**, **660 validated generation logs**, **3300 MAP archive matches**, **270 calibration prompts per model**. Artifacts: `plan_final/`; older `plan_check/` is a development snapshot.
- Eight targeted tests passed (`tests.log`), including actual inventory validation, duplicate/torn/mixed resume rejection, retokenization/serialization tamper detection, correct empty-middle stopping, metric checks against independent DP, and partial-design guards.
- Independent review: `independent_review.json`, no blocker. Reviewer separately tested four original-reconstructor branches with mocked selected outputs and actual pinned tokenizers.
- Aggregate-only execution with CUDA discovery and model loading explicitly forbidden returned the expected partial status and exit code 2 on an empty output (`aggregate_only_no_gpu_check.json`). This check preceded the final extra encoder-validation checks; those checks were subsequently exercised by the targeted tests.
- Python compilation and `git diff --check` passed. No secondary GPU inference was started during preparation.

Do not edit the script or its scientific dependencies after the queued campaign begins. Changed dependencies require a new output directory and protocol.
