# EOS-corrected campaign runners

The runners execute the requested full 6600-record T5 campaign using the completed original inputs. No victim responses are generated again. All source references, length vectors, calibrated inverse vectors, transformation seeds and decoder seeds remain fixed.

## Execution interface

- `inference/rerun_canonical.py`: archived main 2100 records or one of the two fresh-reference cohorts (300 each), `--input`, `--output`, `--device cuda`, `--threads 4`, optional `--limit` and `--plan`.
- A different main `--input` path is accepted only as a diagnostic subset whose **entire rows** exactly match the original canonical inventories. This supports a mixed three-record smoke test. Its protocol explicitly sets `diagnostic_subset=true`; completion means completion of that supplied subset, not the 2700 main/fresh design.
- `secondary/rerun_secondary.py`: 240 controls, 360 defenses and 3300 MAP records. It loads the exact frozen input/calibration inventory instead of fitting a new MAP inverse. `--limit` permits a separately identified partial smoke run; `--require-complete` rejects partial output as a successful full run.
- Both runners reject output paths within the old `revalidation_20261031/`, `results/` or `analysis/` trees. Locks prevent duplicate writers. Do not reuse the preparation-only `secondary/plan_validation/` directory as a production run.

The same checkpoint hashes, software versions and four-thread setting are required. The main/fresh protocol includes all original source bindings and hashes of its new runner, evidence helper and corrected reconstructor. Secondary verifies every original dependency; the old selector dependency is checked against `before/weiss_reconstruction.py`, then the new source map separately binds that snapshot and the corrected root selector. Original protocols and outputs are also bound.

## Candidate evidence and memory

Every `actual_encoder_inputs[k]` contains model ID, sampling count, complete and retained encoder IDs, input text, truncation metadata, `selected_text` and a single `candidate_audit` dictionary. The latter uses `eos-bounded-candidate-audit/1` and retains every candidate's generated IDs, decoded text, normalized transition scores, first EOS position, scored-token count, confidence, eligibility, deduplicated order and selected index. Nonfinite tail values use explicit strings; no numeric sentinel is substituted.

The one-candidate MAP path preserves the historical generation arguments without requesting scores. It still records generated IDs and EOS position. The runner enforces exact equality to each original MAP prediction before committing the new record, making all 3300 records an execution negative control.

JSONL writes are flushed and fsynced per record. A record digest covers all candidate evidence. Resume streams and validates every existing JSONL record; it never silently discards an incomplete or invalid trailing line. Candidate sums and ranks are checked before commit and on resume. Confidence checks allow explicitly bounded float32-versus-Python reduction rounding (`rel_tol=1e-6`, `abs_tol=1e-4`). The separate campaign validator independently checks decoded candidate IDs and context as well.

Completed records are kept in RAM only as minimal aggregate fields after validation: main retains label and φ; secondary retains the fields used by its cell/contrast aggregator. Large candidate arrays remain in JSONL. This avoids retaining the full campaign's candidate tensors as nested Python objects while both T5 models occupy GPU memory.

Each new record also contains the original selector's prediction and φ. For main/fresh, `original_selector_record_sha256` hashes the entire old record because that format had no record hash. For secondary, it is the old record's stored hash (which excludes its own hash field). New record hashes in both formats exclude their own `record_sha256` field.

## Preparation checks

No GPU inference was run by the runner-implementation agent.

- Python syntax compilation: PASS for both runners and their helper.
- Full main inventory plan: 2100 records, seven models, all frozen row hashes verified.
- Full secondary inventory plan: 3900 records; original inventory, calibration, source hashes and per-item digests verified; no refitting.
- Three CPU calls to the real corrected selector plus evidence validation: PASS for post-EOS PAD, EOS-disabled generation and the one-candidate path.
- Mixed frozen-source three-record smoke inventory plan: PASS.
- Deliberately changing a smoke row's decoder seed: rejected before inference.

Code hashes frozen before the root agent's GPU smoke:

| File | SHA-256 |
|---|---|
| `inference/rerun_canonical.py` | `738bb86cd69589afdf8df23215a675b4e6ffd2e2595e301f1a12d6f9b4519165` |
| `inference/candidate_evidence.py` | `9501198e20de8a63122b0e60f2c8fff2977b220e131e3b12a02351f1acc55d5f` |
| `secondary/rerun_secondary.py` | `50c2575c5059b619ef86df73ba4ed57ce717e982fc884c61b126378c650f1fdd` |
