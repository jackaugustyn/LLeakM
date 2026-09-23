# Reproducibility Package

Generated (UTC): 2026-09-19T09:47:51.860916+00:00
Git commit: `eef1f7b917a8557bd6601d0775ffc173a39545a7`

## Environment

- Collection Python: 3.12.3
- Collection platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Analysis re-run Python: 3.10.12
- Analysis re-run platform: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35
- PYTHONHASHSEED: `0`
- CPU: 12th Gen Intel(R) Core(TM) i7-12700H (20 threads)
- Memory: 7.61 GiB
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU (driver 610.62, 8192 MiB, compute 8.6)
- Collection packages:
  - numpy: 2.4.3
  - torch: 2.11.0+cu130
  - transformers: 5.3.0
  - sentence-transformers: not installed
  - scipy: not installed
  - scikit-learn: not installed
  - requests: 2.32.5
  - fastapi: 0.135.2
  - uvicorn: 0.42.0
- Analysis re-run packages:
  - numpy: 2.2.6
  - torch: 2.5.1
  - transformers: 4.46.3
  - sentence-transformers: 3.3.1
  - scipy: 1.8.0
  - scikit-learn: 1.2.2
  - requests: 2.30.0
  - fastapi: 0.128.0
  - uvicorn: 0.40.0
  - protobuf: 4.23.0
  - sentencepiece: not installed

## Seeds

- Analysis / bootstrap seed: `20260706`
- Bootstrap resamples: `10000`
- Victim decoding: temperature `0.0`, top_p `1.0`

The robustness extension uses decoder seed indices `20260706` through
`20260710`. For each prompt and seed index,
`blake2s("decoder", seed_index, prompt_idx)` produces one condition-independent
PyTorch/NumPy/Python seed. Thus baseline, defended, frozen-attacker, and
adaptive-attacker reconstructions use common random numbers. Randomized-padding
noise uses a separate recorded trace seed and remains fixed across decoder
repetitions.

## Model and tokenizer revisions

Local Hugging Face snapshot hashes are recorded when the checkpoint is still in the cache. `not_in_local_cache` means the collection machine no longer holds that snapshot; the Hub identifier remains the publication pin.

| Model | Snapshot | Tokenizer class | Source |
|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | Qwen2Tokenizer | not_in_local_cache |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Qwen2Tokenizer | not_in_local_cache |
| `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | PreTrainedTokenizerFast | not_in_local_cache |
| `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | GemmaTokenizer | not_in_local_cache |
| `microsoft/phi-1_5` | `77aa61eeac94fbf33d492b9f2744c98b42d5b5eb` | CodeGenTokenizer | huggingface_hub_cache |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` | LlamaTokenizer | huggingface_hub_cache |
| `microsoft/Phi-3.5-mini-instruct` | `2fe192450127e6a83f7441aef6e3ca586c338b77` | LlamaTokenizer | huggingface_hub_cache |
| `royweiss1/T5_FirstSentences` | `057cf090b719f5cf89da504babd2122d404e0953` | T5Tokenizer | huggingface_hub_cache |
| `royweiss1/T5_MiddleSentences` | `07e2b3fc96bc129eb4a2871e58731c07e889cecc` | T5Tokenizer | huggingface_hub_cache |

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
- **stats**: `PYTHONHASHSEED=0 python experiment_validation/scripts/analyze_stats.py`
- **extras**: `PYTHONHASHSEED=0 python experiment_validation/scripts/analyze_extras.py`
- **defenses**: `HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 python experiment_validation/scripts/defense_eval.py --per-topic 2 --samples-per-segment 3 --max-sentences 3`
- **robustness plan**: `python experiment_validation/scripts/robustness_eval.py --plan`
- **repeated/adaptive evaluation**: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python experiment_validation/scripts/robustness_eval.py`
- **complete-response collection**: `python experiment_validation/scripts/run_model_matrix.py --models-file experiment_validation/models_publication.json --prompts 300 --max-new-tokens 512 --samples-per-segment 3 --max-sentences 0 --num-first-candidates 3 --prompt-format legacy --reconstruction-seed 20260706 --hf-offline`
- **figures**: `python experiment_validation/scripts/make_figures.py`
- **audit**: `python experiment_validation/scripts/audit_release_content.py`
- **verify**: `python experiment_validation/scripts/verify_manuscript_numbers.py`
- **manifest**: `python experiment_validation/scripts/make_repro_manifest.py`

## Robustness-extension safeguards

- The default robustness test split contains the first two aligned prompts per
  topic; all remaining prompts are calibration-only.
- The adaptive MAP inverse is fitted independently for each victim model and
  defense without using test responses.
- `max_sentences=0` means every segment found by the reconstructor is decoded.
- Complete-response claims require `response_complete=true`, which is emitted
  only when victim generation reaches EOS before the token cap.
- `robustness_eval.py` emits a LaTeX table only after every expected
  prompt-by-seed cell is present. Until then, `complete` remains `false`.

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
