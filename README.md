# LLeakM — LLM Output Reconstruction from Token-Length Traces

LLeakM is a controlled research harness and reproducibility package for studying how much generated LLM text can be reconstructed from token-length traces exposed by token-by-token Server-Sent Events (SSE) streaming. The repository contains the streaming server, a Weiss-style T5 reconstructor, a 300-prompt benchmark, results for seven open-weight victim models, statistical analyses, defense experiments, and IEEE/MDPI manuscript sources.

## Scope and threat model

The benchmark uses an instrumented server that emits exactly one generated token per SSE frame and records its UTF-8 length. This is an ideal token-length oracle: it isolates the information in the length sequence, but it is not an end-to-end attack on TLS, HTTP/2, or QUIC traffic. Real encrypted transports may coalesce, split, or pad application frames.

The reconstructor is fixed across models and defenses. Reported differences therefore measure transfer of this particular Weiss-style reconstructor, not an intrinsic, attacker-independent leakage ranking. Defense results use a non-adaptive attacker that was not retrained on defended traces.

## Current benchmark

The primary dataset contains 300 aligned prompts from 15 topics for each of seven models (2100 samples). Generation is deterministic (`temperature=0`, `top_p=1`) with a 96-token cap. Reconstruction uses three samples per segment, at most three sentences, and three first-sentence candidates.

The final analysis recomputes a single term-frequency cosine score `phi` for every sample. It is a lexical-overlap metric, not a semantic embedding score. `LR@0.5` is the percentage of samples with `phi > 0.5`.

| Victim model | Kind | Mean `phi` | 95% CI | `LR@0.5` |
|---|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | instruct | 0.3242 | [0.3097, 0.3380] | 8.67% |
| Qwen2.5-3B-Instruct | instruct | 0.3096 | [0.2977, 0.3214] | 3.00% |
| Phi-1.5 | base | 0.2747 | [0.2622, 0.2877] | 4.33% |
| Llama-3.2-3B-Instruct | instruct | 0.2538 | [0.2419, 0.2658] | 3.00% |
| Gemma-2-2B-it | instruct | 0.1798 | [0.1708, 0.1889] | 0.00% |
| TinyLlama-1.1B-Chat | chat | 0.0428 | [0.0377, 0.0479] | 0.00% |
| Phi-3.5-mini-instruct | instruct | 0.0239 | [0.0201, 0.0280] | 0.00% |

Bootstrap confidence intervals, paired Wilcoxon tests, Holm correction, response-length controls, trace-information baselines, and defense results are in `experiment_validation/analysis/`. Nineteen of 21 cross-model comparisons remain significant after Holm correction.

## Repository layout

```text
.
├── app.py                              # one-token-per-frame SSE server and JSON logging
├── client.py                           # example streaming client
├── weiss_reconstruction.py             # Weiss-style T5 reconstruction
├── experiment_validation/
│   ├── prompts/                        # 300 prompts, 15 topics
│   ├── results/                        # per-run samples and summaries
│   ├── scripts/                        # collection, analysis, baselines, defenses
│   └── analysis/                       # final tables, statistics, figures, manifest
├── article1/
│   ├── tifs_main.tex / tifs_main.pdf   # IEEE TIFS version
│   └── template.tex / template.pdf     # MDPI version
├── report_LLeakM.md                    # current experimental report
└── experiment.txt                      # concise current experiment specification
```

The seven runs used by the manuscript are listed in `experiment_validation/analysis/repro_manifest.json`. Additional tracked runs are historical and are not used for the primary claims.

## Installation

Python 3.10 or newer is required. CUDA is strongly recommended; the server falls back to CPU, but generation and T5 reconstruction are then substantially slower.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Model weights are loaded from Hugging Face. Set `HF_HUB_OFFLINE=1` only after all required checkpoints are cached locally.

## Run the SSE harness

Start the server with the default Qwen2.5-0.5B victim model:

```bash
. .venv/bin/activate
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Select another victim model and log directory with environment variables:

```bash
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct LOG_DIR=./logs \
  python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
. .venv/bin/activate
python client.py
python weiss_reconstruction.py
```

`client.py` prints observed SSE frame lengths. `weiss_reconstruction.py` without an argument performs a new live request and reconstructs it. To replay a stored trace, pass its run ID:

```bash
python weiss_reconstruction.py <run-id>
```

The endpoint is `GET /generate_sse` with `prompt`, `max_new_tokens`, `temperature`, and `top_p` query parameters. Each completed request is stored as `logs/<run-id>.json` with the prompt, response, model metadata, and per-token lengths and timestamps.

## Reproduce the benchmark and analysis

Detailed commands and the exact standardized run mapping are in [`experiment_validation/README.md`](experiment_validation/README.md). A single-model collection uses:

```bash
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_validation.py \
  --max-prompts 300 \
  --max-new-tokens 96 \
  --samples-per-segment 3 \
  --max-sentences 3 \
  --num-first-candidates 3
```

Regenerate the final analysis from the tracked per-sample results:

```bash
python experiment_validation/scripts/analyze_stats.py
python experiment_validation/scripts/analyze_extras.py
python experiment_validation/scripts/make_figures.py
python experiment_validation/scripts/make_repro_manifest.py
```

The defense and trace-baseline scripts require the 60 tracked raw traces for the topic-balanced Qwen2.5-1.5B and Llama-3.2-3B subsets. Other runtime logs are intentionally ignored.

## Build the manuscripts

```bash
cd article1
latexmk -pdf tifs_main.tex
latexmk -pdf template.tex
```

Generated LaTeX intermediates, `.venv`, caches, non-manuscript runs, and ordinary runtime logs are excluded by `.gitignore`. Final manuscript PDFs, standardized per-sample artifacts, analysis outputs, and the required 60 defense/baseline traces are versioned.

## Reproducibility and limitations

- Human-readable protocol and environment: `experiment_validation/analysis/REPRODUCIBILITY.md`
- SHA-256 manifest: `experiment_validation/analysis/repro_manifest.json`
- Full current report: `report_LLeakM.md`
- Manuscript: `article1/tifs_main.tex`

Nearly all responses hit the 96-token cap, so the benchmark characterizes early-response reconstructability. The fixed lexical metric can reward shared wording even when meaning or entities differ. Live ciphertext extraction, adaptive attackers trained on defended traces, longer generations, multiple reconstruction seeds, and human semantic evaluation remain future work.
