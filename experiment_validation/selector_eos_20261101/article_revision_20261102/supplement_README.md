# LLeakM-S2-EOS-20261102-v1

This local supplement freezes the article and the **complete new EOS-inclusive reconstruction campaign**, completed on 2 November 2026. All reconstruction estimates, conclusions, numerical tables and generated result figures in the enclosed manuscript use these new outputs. The package has not been deposited in a public repository and its identifier is not a DOI.

The article is `article1/template.tex` / `article1/template.pdf`. Its data, table and figure generators are under `experiment_validation/selector_eos_20261101/article_revision_20261102/`; its result figures have the `article1/figures/eos_` prefix. S1 and old manuscript figures are not part of this package.

## New result inventory

All paths in this table are relative to `experiment_validation/selector_eos_20261101/`.

| Retained output | Records | Scope |
|---|---:|---|
| `inference/cuda_full/samples.jsonl` | 2100 | Seven fixed reference series, including the two diagnostic rendering series |
| `inference/fresh_tinyllama_cuda/samples.jsonl` | 300 | TinyLlama verified reference rendering |
| `inference/fresh_phi_cuda/samples.jsonl` | 300 | Phi-3.5 verified reference rendering |
| `secondary/cuda_full/samples.jsonl` | 3900 | 240 trace controls, 360 defense records and 3300 MAP records |
| **Total** | **6600** | New reconstruction outputs only |

Every new reconstruction retains all attempted T5 encoder contexts and candidate evidence. There are 86,788 generation calls, including 83,488 calls with three candidates and 3300 calls with one candidate, and 253,764 candidate sequences in total. Candidate token IDs, first-EOS positions and exact selected indices are retained for every call. Processed transition scores are retained for the 83,488 calls with three candidates. The 3300 single-candidate MAP calls bypass confidence ranking and store `transition_scores: null`, `scored_token_count: 0` and a zero confidence sentinel. The 4,540,261 scored transition positions end at and include the first EOS when one is present; 393,353 post-EOS positions are excluded. String encodings such as `"-Infinity"` describe saved scores outside the scoring prefix and are not silently converted into finite values.

New protocols, frozen input inventories, pinned checkpoint revisions and hashes, metric recomputations and statistical audits are included alongside the new outputs. The corrected implementation is root `weiss_reconstruction.py`; the regression tests and independent candidate audit are included. Small T5 configuration and tokenizer files are bundled under `article_revision_20261102/pinned_tokenizers/`, with exact upstream revisions, source URLs and SHA-256 values. Model weights are excluded.

## Frozen reference inputs and provenance

The new campaign reconstructs existing, fixed references and length vectors. It does **not** claim that the 2700 source responses were generated again. The 600 verified TinyLlama/Phi references retain their victim token IDs and generation logs from the earlier reference-generation step. Their fixed-ID rendering audit is reproduced in this revision. The main diagnostic series still has the documented reference/rendering limitations; it is not relabeled as clean model-level evidence.

Necessary raw references, token lengths, prompts, source logs, calibration inputs and generation metadata retain their original paths under `experiment_validation/revalidation_20261031/`, `experiment_validation/results/` and `logs/`. Some new protocols bind earlier reconstruction files by hash, and the immutable new records include `old_phi`, `original_selector_*` and `legacy_full_step_confidence` fields. Those archived files and fields are preserved solely as source provenance. They are **not** numerical evidence for this manuscript, are not included in its figure/table calculations, and are not used by the portable verifier to compare new predictions with old predictions. The new 3300 MAP records are verified from their own inputs, candidate records and lexical scores.

The package includes the model-card source extraction JSON used to identify the pinned checkpoint training scope. It does not redistribute full third-party papers, model weights, caches, old review narratives, previous PDFs, old result figures, partial campaigns or S1. Retained campaign audit reports may record historical comparisons as provenance; the current article’s numerical source ledger and portable checks concern only the new outputs.

## Verify without model inference

From the extracted package root, use Python 3.10 or later:

```sh
python3 verify_supplement.py
```

This command needs only Python's standard library and does not modify files. It checks every inventoried file's byte length and SHA-256; complete model/condition/seed grids; all 6600 reconstruction record digests and input bindings; 2760 distinct source logs and 600 recorded victim-ID sequences; all 6600 bag-of-words cosine values; segmentation and context serialization; all EOS-prefix score sums, candidate eligibility/deduplication/ranking/ties; and final response assembly. Saved scores are CUDA float32 reductions, so independently accumulated float64 prefix sums use the explicit bound `max(1e-5, 2e-6 * max(1, sum(abs(prefix_scores))))`. Ranking and tie checks are exact against the retained float32 confidences.

To additionally re-encode every encoder context and decode every candidate sequence offline, install `tokenizers==0.20.3` in your own environment and run:

```sh
python3 verify_supplement.py --tokenizers
```

This additional mode uses only the bundled, hash-verified tokenizer/configuration files. It checks all 86,788 encoder inputs and 253,764 decoded candidates; it does not download anything. The default standard-library mode does not independently establish token-ID/string equivalence. Neither mode runs T5 forward passes, regenerates victim responses, evaluates semantic disclosure, verifies live network captures or certifies current journal compliance.

The retained independent statistical scripts and new table/figure generators can recompute the reported analyses from these fixed records. `requirements-cpu.txt` records the Python 3.10 analysis packages used for NumPy/SciPy/Matplotlib analyses; it is unnecessary for the standard-library verifier. The full candidate audit additionally uses the pinned Transformers runtime. Some campaign scripts use the local Hugging Face cache and require checkpoint weights that are deliberately absent here. Running them may write their own report files; work on a copy when preserving this frozen package.

## Inventory and versioning

`MANIFEST.json` records the exact path, byte length and SHA-256 of every content file. It excludes itself and `SHA256SUMS` to avoid a hash cycle. `SHA256SUMS` binds both the manifest and every content file, excluding itself. The external `.tar.gz.sha256` binds the complete compressed archive; the external `.manifest.json` and `.verification.json` permit inspection before extraction. Later workspace revision manifests, package-dependent validation outputs and temporary build/render directories are excluded.

The builder sorts paths and fixes gzip/tar timestamps, ownership and permissions. It rejects source changes detected during writing, extracts the completed archive into a separate temporary directory, and runs the portable verifier against that extracted copy before promoting it to the versioned destination. Existing package IDs cannot be overwritten with different bytes. All artifacts remain local unless explicitly deposited later.
