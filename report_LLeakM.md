# LLeakM Experimental Report

- **Updated:** 2026-08-03
- **Status:** publication package complete
- **Primary sample:** 7 victim models x 300 aligned prompts = 2100 reconstructions

## 1. Executive summary

LLeakM evaluates reconstruction of generated LLM output from token-length traces in controlled, token-by-token Server-Sent Events (SSE) streaming. An instrumented server emits one generated token per SSE frame and records its UTF-8 length. A shared Weiss-style T5 pipeline reconstructs candidate text from the resulting length sequence.

This is an idealized length-oracle study, not an end-to-end extraction attack on TLS, HTTP/2, or QUIC traffic. The experiment deliberately separates the information available in token lengths from transport-layer complications such as record coalescing, fragmentation, padding, and multiplexing.

The final standardized benchmark contains seven open-weight victim models and 2100 samples. Six models are instruction/chat-tuned; Phi-1.5 is retained as a base-model stress test. All models share the same prompts, generation budget, reconstructor, and final lexical metric.

Key findings:

1. Mean lexical reconstruction score `phi` varies from 0.3242 for Qwen2.5-1.5B-Instruct to 0.0239 for Phi-3.5-mini-instruct.
2. Nineteen of 21 paired cross-model differences remain significant after Holm correction.
3. Model identity retains explanatory power after controlling for response length, mean token bytes, and truncation (`R²=0.646` for the full model versus `0.642` for model indicators alone and `0.511` for covariates alone).
4. Real token-length traces outperform shuffled, constant-length, and cross-prompt controls on the two-model baseline subset.
5. Five defense configurations substantially reduce reconstruction score against the fixed, non-adaptive reconstructor. No defended sample exceeds `phi > 0.5` in the evaluated subset, although the Wilson upper bound remains about 11.35%.

These results measure transferability of a fixed reconstructor across models and trace transformations. They do not establish an intrinsic model leakage ranking or robustness against an adaptive attacker trained on defended traces.

## 2. Reproducibility package status

The publication package contains:

- IEEE TIFS and MDPI manuscript sources and compiled PDFs;
- 300 prompts across 15 topics;
- six artifacts for each of the seven standardized model runs;
- per-sample ground truth, reconstructions, and metrics;
- final cross-model statistics, confidence intervals, pairwise tests, covariate controls, and regression outputs;
- trace-information baselines and defense outputs;
- 60 raw traces required by the topic-balanced baseline/defense subsets;
- a machine-readable SHA-256 manifest and environment snapshot.

Artifact integrity is recorded in `experiment_validation/analysis/repro_manifest.json`. The human-readable companion is `experiment_validation/analysis/REPRODUCIBILITY.md`. Additional tracked model runs are historical and are excluded from primary claims.

## 3. Threat model and measurement harness

The attacker observes the length sequence associated with a streamed model response and attempts to infer the generated text. In the controlled harness:

- the application emits `data: <token>\n\n` once per generated token;
- the terminal `data: [DONE]\n\n` frame is excluded from reconstruction;
- token length is recovered as `sse_frame_utf8_len - 8` bytes;
- logs contain the prompt, ground-truth response, model configuration, token text, token UTF-8 length, frame length, and relative timestamp.

Because application framing is known and one token maps to one frame, this setup is optimistic for the attacker relative to live ciphertext-only observation. A real network attacker must first recover an equivalent sequence from encrypted transport records.

## 4. Reconstruction pipeline

`weiss_reconstruction.py` implements a fixed Weiss-style sequence-to-text pipeline:

1. Convert SSE frame lengths to token UTF-8 lengths.
2. Segment the sequence with the fixed length-based sentence heuristic.
3. Decode the first segment with `royweiss1/T5_FirstSentences`.
4. Decode later segments with `royweiss1/T5_MiddleSentences` using reconstructed context.
5. Rank sampled candidates by generation confidence and concatenate the selected hypotheses.

The same reconstructor and hyperparameters are used for every victim model. The ranking below therefore reflects cross-model transfer under one attacker configuration, not the best achievable per-model attack.

## 5. Dataset and standardized protocol

The prompt corpus consists of 15 topical files with 20 prompts each. Topics cover health, mental health, sexual health, substance use, legal and financial issues, employment, education, relationships, family planning, identity, self-esteem, safety, beliefs, and personal health.

Standardized settings:

| Parameter | Value |
|---|---:|
| Prompts per model | 300 |
| Victim models | 7 |
| Total samples | 2100 |
| `max_new_tokens` | 96 |
| `temperature` | 0.0 |
| `top_p` | 1.0 |
| Reconstruction samples per segment | 3 |
| Maximum reconstructed sentences | 3 |
| First-sentence candidates | 3 |
| Analysis seed | 20260706 |

Nearly all responses reach the tight 96-token budget (approximately 95–100% depending on model), so the study characterizes early-response reconstructability rather than complete long-form responses.

## 6. Metrics and statistics

### 6.1 Final lexical metric

The final comparison recomputes a term-frequency cosine score `phi` between bag-of-words vectors of the ground truth and reconstruction. It measures lexical overlap and is not a paraphrase-aware semantic embedding metric. High `phi` can coexist with changed entities or meaning.

`LR@0.5` is the fraction of samples with `phi > 0.5`. The threshold is conventional and has not been calibrated against human judgments.

Supporting metrics are normalized character-level Levenshtein distance (`ed_norm`) and ROUGE-1/L precision.

Historical `phi_cosine` fields in individual run summaries may reflect the backend selected when that run was collected. Final publication values come from `analysis/per_prompt_phi.csv` and `analysis/model_stats.json`, where every model is rescored with the same term-frequency backend.

### 6.2 Inference

- Mean-score uncertainty: 10,000-sample bootstrap 95% confidence intervals.
- `LR@0.5`: Wilson 95% confidence intervals.
- Cross-model comparisons: paired Wilcoxon signed-rank tests over aligned prompts.
- Multiple comparisons: Holm–Bonferroni correction over 21 model pairs.
- Effect size: rank-biserial correlation.
- Covariate analysis: OLS with model indicators, response characters, mean token bytes, and truncation.

## 7. Main cross-model results

| Model | Kind | N | Mean `phi` | 95% CI | `ed_norm` | R1 precision | RL precision | `LR@0.5` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | instruct | 300 | 0.3242 | [0.3097, 0.3380] | 0.7199 | 0.3220 | 0.2651 | 8.67% |
| Qwen2.5-3B-Instruct | instruct | 300 | 0.3096 | [0.2977, 0.3214] | 0.7300 | 0.2945 | 0.2345 | 3.00% |
| Phi-1.5 | base | 300 | 0.2747 | [0.2622, 0.2877] | 0.7426 | 0.2503 | 0.1891 | 4.33% |
| Llama-3.2-3B-Instruct | instruct | 300 | 0.2538 | [0.2419, 0.2658] | 0.7499 | 0.2387 | 0.1824 | 3.00% |
| Gemma-2-2B-it | instruct | 300 | 0.1798 | [0.1708, 0.1889] | 0.7764 | 0.1719 | 0.1269 | 0.00% |
| TinyLlama-1.1B-Chat | chat | 300 | 0.0428 | [0.0377, 0.0479] | 0.8026 | 0.0312 | 0.0310 | 0.00% |
| Phi-3.5-mini-instruct | instruct | 300 | 0.0239 | [0.0201, 0.0280] | 0.8087 | 0.0158 | 0.0157 | 0.00% |

Phi-1.5 is not directly comparable as an instruction-following assistant and is excluded from the primary instruction/chat-only interpretation. Its relatively high lexical score demonstrates that the fixed reconstructor can transfer to base-model output style; it should not be interpreted as evidence that base models are inherently more vulnerable.

### 7.1 Pairwise tests

Nineteen of 21 paired model comparisons are significant at `p_Holm < 0.05`. The two non-significant comparisons are:

- Qwen2.5-1.5B-Instruct versus Qwen2.5-3B-Instruct (`p_Holm=0.1838`);
- Llama-3.2-3B-Instruct versus Phi-1.5 (`p_Holm=0.0533`).

### 7.2 Length and truncation controls

For the six instruction/chat models (`n=1800`):

| Regression | R² |
|---|---:|
| Full model + length controls | 0.646 |
| Model indicators only | 0.642 |
| Length/truncation covariates only | 0.511 |

The full model improves only slightly over model identity alone, while covariates alone explain substantially less variance. The observed ordering is therefore not reducible to response length statistics, although tokenizer and style remain part of what the fixed reconstructor transfers across.

## 8. Trace-information baselines

The baseline study uses 30 topic-balanced prompts for each of Qwen2.5-1.5B-Instruct and Llama-3.2-3B-Instruct. It compares the real trace with three degraded controls.

| Model | Trace | N | Mean `phi` | Change vs real |
|---|---|---:|---:|---:|
| Qwen2.5-1.5B | real | 30 | 0.3024 | — |
|  | shuffled | 30 | 0.2091 | -0.0933 |
|  | constant | 30 | 0.1068 | -0.1957 |
|  | cross-prompt | 30 | 0.2561 | -0.0463 |
| Llama-3.2-3B | real | 30 | 0.2913 | — |
|  | shuffled | 30 | 0.1872 | -0.1041 |
|  | constant | 30 | 0.1242 | -0.1671 |
|  | cross-prompt | 30 | 0.2312 | -0.0601 |

Real traces outperform every degraded condition. Cross-prompt traces remain closer to the real condition than shuffled or constant traces, indicating that both sequence structure and the reconstructor's language prior contribute to recovery.

## 9. Defense evaluation

Five configurations are evaluated against the same undefended-trained reconstructor:

- `bucket_8`: round lengths up to multiples of 8;
- `pad_32`: emit an online fixed frame width of 32 bytes;
- `batch_2` and `batch_4`: combine two or four tokens per observable frame;
- `rand_pad_8`: add seeded random padding from 0 to 8 bytes.

Each model/defense cell contains 30 topic-balanced prompts.

| Model | Defense | Mean `phi` | Leakage reduction | Mean byte overhead | `LR@0.5` |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | `bucket_8` | 0.0293 | 90.3% | 82.4% | 0.0% |
|  | `pad_32` | 0.1901 | 36.8% | 538.0% | 0.0% |
|  | `batch_2` | 0.0928 | 69.2% | 0.0% | 0.0% |
|  | `batch_4` | 0.0950 | 68.4% | 0.0% | 0.0% |
|  | `rand_pad_8` | 0.1033 | 65.7% | 81.6% | 0.0% |
| Llama-3.2-3B | `bucket_8` | 0.0428 | 83.4% | 88.6% | 0.0% |
|  | `pad_32` | 0.1674 | 35.0% | 565.1% | 0.0% |
|  | `batch_2` | 0.0823 | 68.0% | 0.0% | 0.0% |
|  | `batch_4` | 0.0832 | 67.7% | 0.0% | 0.0% |
|  | `rand_pad_8` | 0.1004 | 61.0% | 85.0% | 0.0% |

All ten defense-versus-baseline comparisons remain significant after Holm correction. No defended sample exceeds `phi > 0.5`; with only 30 samples per cell, the Wilson upper bound is approximately 11.35%, so zero observed successes must not be interpreted as proof of zero risk.

`bucket_8` is strongest against this fixed reconstructor. `pad_32` is weaker despite very high byte overhead because a constant length far above ordinary token size still induces a reconstruction prior. Batching reduces observable frame count with zero payload-byte-sum overhead under the study's accounting, but transport and latency costs are not modeled.

## 10. Interpretation and limitations

The benchmark supports the following narrow conclusion: token-length sequences contain usable lexical information for a fixed Weiss-style reconstructor, and transfer quality differs substantially across victim-model outputs under controlled observation.

The results do not establish:

- extraction feasibility from live encrypted network records;
- an attacker-independent ranking of victim models;
- semantic correctness of high-`phi` reconstructions;
- security against a reconstructor retrained on each model or defense;
- behavior for long, untruncated responses;
- robustness across multiple reconstruction seeds.

The defense study is a pilot against a frozen attacker and should be treated as evidence of mitigation potential, not a security proof.

## 11. Remaining work

1. Recover application-length observations from real TLS/HTTP2/QUIC captures.
2. Train adaptive attackers on padded, bucketed, and batched traces.
3. Repeat reconstruction with multiple seeds and report variance or best-of-k behavior.
4. Extend generation budgets to 256–384 tokens and analyze completed responses.
5. Add BERTScore or another fixed embedding metric together with blinded human judgments.
6. Measure latency, bandwidth, and user-experience costs of defenses at transport level.

## 12. Authoritative files

- Manuscript: `article1/tifs_main.tex`
- Standardized run manifest: `experiment_validation/analysis/repro_manifest.json`
- Model-level results: `experiment_validation/analysis/model_stats.json`
- Per-prompt final `phi`: `experiment_validation/analysis/per_prompt_phi.csv`
- Pairwise tests: `experiment_validation/analysis/pairwise_tests.json`
- Covariate regression: `experiment_validation/analysis/covariate_regression.json`
- Trace baselines: `experiment_validation/analysis/trace_baselines.json`
- Defense results: `experiment_validation/analysis/defense_eval.json`
- Reproduction guide: `experiment_validation/analysis/REPRODUCIBILITY.md`

## References

1. Weiss, R., Ayzenshteyn, D., Amit, G., and Mirsky, Y. *What Was Your Prompt? A Remote Keylogging Attack on AI Assistants.* USENIX Security, 2024.
2. Raffel, C. et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR, 2020.
3. Lin, C.-Y. *ROUGE: A Package for Automatic Evaluation of Summaries.* ACL Workshop, 2004.
4. Levenshtein, V. I. *Binary Codes Capable of Correcting Deletions, Insertions and Reversals.* 1966.
5. Efron, B., and Tibshirani, R. J. *An Introduction to the Bootstrap.* 1994.
6. Wilcoxon, F. *Individual Comparisons by Ranking Methods.* 1945.
7. Holm, S. *A Simple Sequentially Rejective Multiple Test Procedure.* 1979.
