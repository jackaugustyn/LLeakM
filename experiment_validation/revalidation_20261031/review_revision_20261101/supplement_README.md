# Supplementary Data S1: LLeakM-S1-20261101-v1

This version contains the manuscript, code, prompts, source logs, completed predictions, numerical checks and plotting inputs for the 1 November 2026 revision. It is a frozen **local supplementary package prepared for submission**. Its presence here does not establish a public release, DOI or journal deposit. The source project is https://github.com/jackaugustyn/LLeakM; public archival of this completed package remains pending.

The quantitative campaign contains 6600 T5 reconstructions: 2100 from seven archived-reference cohorts, 600 from two newly generated and validated reference cohorts, 240 partial trace ablations, 360 framing-defense trials, and 3300 first-segment MAP trials. The 600 fresh victim generations retain generated token IDs and complete fragment logs. The bounded candidate-ranking audit in `review_revision_20261101/ranking_audit/` is separate diagnostic evidence and is not included in the 6600 count.

The main tables describe the literal executed T5 implementation. Its multi-candidate selector sums transition scores beyond the first EOS, including padding positions. These results must not be presented as an evaluation of a corrected EOS-bounded selector. The bounded audit documents that distinction; consult the manuscript and its point-by-point revision report for the measured consequences and limits. The one-candidate MAP study performs no ranking among candidates.

## Verify the extracted package

Extract the archive into an empty directory, enter `LLeakM-S1-20261101-v1`, and run:

```bash
python3 verify_supplement.py
```

Python 3.10 or later is sufficient. No GPU, network connection, third-party Python package or model weight is required. The verifier does not modify files. It checks:

- SHA-256 and byte count of every manifest member, and the binding of `MANIFEST.json` by `SHA256SUMS`;
- complete and unique keys for all 6600 reconstructions, including every secondary condition and seed;
- agreement with the retained canonical/input inventories and source logs;
- reference hashes, UTF-8 fragment lengths and concatenation, and all 600 fresh generated-ID records;
- term-frequency cosine independently recalculated from each saved prediction and reference, at absolute tolerance `1e-12`.

A PASS verifies these retained records and calculations. It does not perform model inference, independently validate a model's training, establish semantic disclosure, estimate channel capacity, or certify passive extraction from encrypted traffic. Manifest hashes detect changes relative to this package, not the historical truth of a record.

The adjacent `LLeakM-S1-20261101-v1.tar.gz.sha256` file checks the compressed archive. An identical external copy of its inventory is `LLeakM-S1-20261101-v1.manifest.json`. Inside the archive, `MANIFEST.json` inventories every content file except itself and `SHA256SUMS`; `SHA256SUMS` binds all inventoried files plus the manifest, excluding itself. This avoids a circular digest. The package builder uses fixed tar metadata and gzip timestamps. Its later package-dependent revision manifest is deliberately outside this archive.

## Locate the evidence

All paths below are relative to this extracted directory. The project directory structure is preserved.

| Evidence | Location |
|---|---|
| Revised manuscript, bibliography and figures | `article1/` |
| Archived source responses and selected raw logs | `experiment_validation/results/` and `logs/` |
| Validated archived canonical vectors and references | `experiment_validation/revalidation_20261031/collection/canonical_traces.jsonl` |
| 2100 archived-reference reconstructions | `experiment_validation/revalidation_20261031/inference/cuda_full/` |
| 600 fresh generations, token IDs and logs | `experiment_validation/revalidation_20261031/fresh_references/{tinyllama,phi}/` |
| 600 fresh-reference reconstructions | `experiment_validation/revalidation_20261031/inference/{fresh_tinyllama_cuda,fresh_phi_cuda}/` |
| 3900 control, defense and MAP records | `experiment_validation/revalidation_20261031/secondary/cuda_full/` |
| Complete inference verification and statistics | `experiment_validation/revalidation_20261031/inference_validation/` |
| Independent metric/statistical audits | `experiment_validation/revalidation_20261031/final_review_20261101/` |
| Numerical figure sources and generator | `experiment_validation/revalidation_20261031/article_revision_latest_20261101/` |
| Current revision patches and bounded ranking audit | `experiment_validation/revalidation_20261031/review_revision_20261101/` |

Each completed inference directory contains its protocol, sample inventory/output and completion status. All source dependencies listed in `secondary/cuda_full/sources.json` are included with their original hashes. Original source-response directories are included for the canonical cohort and two pilots. Full third-party papers, model weights, download caches, partial main-campaign/probe runs, previous PDF copies and historical whole-workspace inventories are excluded. The ranking audit retains its separately marked two-record `attempt_01` instrumentation failure for provenance; it is excluded from the 24-case denominator. Small pre-revision TeX and BibTeX snapshots are included so that the exact-text patch can be replayed in a working copy. References in archived reports to excluded probes or absolute workstation/cache paths remain historical provenance; they are not instructions to read those paths. `MANIFEST.json` alone defines this package's membership. The published model cards used to verify the documented checkpoint provenance are retained as text in `sources_proposal.json`.

## Repeat numerical analyses and figures

The entry-point verifier above is dependency-free. More extensive audits use NumPy and SciPy; plots also need Matplotlib. `requirements-cpu.txt` records the versions available for these analyses on Python 3.10. The independent main/fresh string-metric audit additionally requires a C++17 compiler (`g++`). LaTeX compilation needs the ordinary packages used by the included MDPI template and TikZ.

The following scripts write fresh reports or plot files at their documented project locations. Run them in a **working copy** of the extracted package, retaining the original extraction for checksum verification:

```bash
python3 -m venv analysis-env
analysis-env/bin/python -m pip install -r requirements-cpu.txt
analysis-env/bin/python experiment_validation/revalidation_20261031/final_review_20261101/audit_main_fresh.py
analysis-env/bin/python experiment_validation/revalidation_20261031/final_review_20261101/secondary_audit.py
analysis-env/bin/python experiment_validation/revalidation_20261031/article_revision_latest_20261101/generate_figures.py
```

These regenerate the retained metric/statistical audits and numeric figures, not model outputs. Report timestamps and plot metadata can differ. The archived scripts may assert checkpoint/code hashes or include original absolute paths; the dependency-free verifier provides a portable initial check. Numeric ledgers and raw predictions are both included, so plotted values can also be independently derived without these scripts.

## Model inference provenance and retrieval

Weights are excluded. The checkpoint protocols record revisions and SHA-256 hashes of downloaded files. Retrieve the exact versions from Hugging Face under their applicable model licenses if new inference is required:

| Role | Model ID | Revision |
|---|---|---|
| T5 first segment | `royweiss1/T5_FirstSentences` | `057cf090b719f5cf89da504babd2122d404e0953` |
| T5 subsequent segments | `royweiss1/T5_MiddleSentences` | `07e2b3fc96bc129eb4a2871e58731c07e889cecc` |
| Fresh TinyLlama victim | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| Fresh Phi-3.5 victim | `microsoft/Phi-3.5-mini-instruct` | `2fe192450127e6a83f7441aef6e3ca586c338b77` |

For example, in an inference environment with `huggingface_hub` installed, this explicitly downloads the first T5 checkpoint:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="royweiss1/T5_FirstSentences",
    revision="057cf090b719f5cf89da504babd2122d404e0953",
)
```

The recorded reconstruction environment used PyTorch `2.5.1+cu124`, Transformers `4.46.3`, CUDA, FP32 T5 weights, four CPU threads, 80-token encoder and decoder limits, top-k sampling with `k=50`, temperature 1 and a two-token no-repeat-ngram constraint. Both T5 checkpoints and all exact per-record seeds are pinned in the protocols. Fresh victims used FP16; Phi used CPU offload, with placement recorded in its runtime file. Historical victim token IDs/revisions for the archived cohort remain unavailable; current checkpoint retrieval does not recover them.

`inference/rerun_canonical.py`, `secondary/rerun_secondary.py`, `fresh_references/generate_fresh.py` and `weiss_reconstruction.py` preserve the completed campaign code. Consult each runner's `--help` and protocol before a new run. Always choose a new output directory: restarting an old completed directory is not an experiment with a new selector. Stochastic outputs across different hardware or software are not promised to be bit-identical.
