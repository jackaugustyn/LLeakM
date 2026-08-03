# Reproducibility Package

Generated (UTC): 2026-08-03T10:22:10.568165+00:00
Git commit: `fb304816988884a7e298e07f9d997b09c7f049a3`

## Environment

- Python: 3.12.3
- Platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- numpy: 2.4.3
- torch: 2.11.0
- transformers: 5.3.0
- sentence-transformers: not installed
- scipy: not installed
- scikit-learn: not installed
- requests: 2.32.5
- fastapi: 0.135.2
- uvicorn: 0.42.0

## Standardized protocol

- N: 300
- max_new_tokens: 96
- samples_per_segment: 3
- max_sentences: 3
- num_first_candidates: 3
- temperature: 0.0
- top_p: 1.0
- semantic_backend: term-frequency cosine (fixed, recomputed for all runs)

## Regeneration commands

- **prompts**: `python experiment_validation/scripts/generate_prompts.py`
- **collect**: `HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_validation.py --max-prompts 300 --max-new-tokens 96 --samples-per-segment 3 --max-sentences 3`
- **stats**: `python experiment_validation/scripts/analyze_stats.py`
- **defenses**: `HF_HUB_OFFLINE=1 python experiment_validation/scripts/defense_eval.py --per-topic 2 --samples-per-segment 3 --max-sentences 3`
- **figures**: `python experiment_validation/scripts/make_figures.py`
- **manifest**: `python experiment_validation/scripts/make_repro_manifest.py`

## Artifact integrity (SHA-256)

Each standardized run directory contains six artifacts; hashes are in `repro_manifest.json` under `artifacts`. Analysis outputs and code are hashed under `analysis_outputs` and `code`.

| Run | samples.jsonl SHA-256 (prefix) | bytes |
|---|---|---:|
| run_20260325_104518_qwen_1_5b | `f8c3e63b59c64a02...` | 523291 |
| run_20260327_124817_qwen_3b_final_resume | `da6ee761f66f1e33...` | 538515 |
| run_20260414_073030_llama_3_2_3b | `6e9da9245028d9d5...` | 522806 |
| run_20260415_080034_gemma_2_2b_it | `c22218c3f3bd5c29...` | 485696 |
| run_20260329_085714_phi_1_5 | `6fac7a82033c75fa...` | 524240 |
| run_20260325_153010_tinyllama_1_1b | `9df32964356282d2...` | 474266 |
| run_20260414_111644_phi_3_5_mini | `eca2972edc41ad8a...` | 480241 |
