# Independent scientific checks for the EOS campaign manuscript

The sole reconstruction evidence for the revised manuscript is the completed `selector_eos_20261101` campaign: 6600 records. Its frozen archived victim responses are legitimate experimental inputs. Neither their archival provenance nor the matched legacy/full-sequence rendering experiment constitutes a comparison with an earlier reconstruction campaign. No old-selector prediction or old-versus-current reconstruction difference should remain in the manuscript.

Exact values, intervals, all primary pairwise tests, all secondary cells and contrasts, source hashes, and independent record-level checks are in `scientific_facts.json` alongside this note.

## Main results and interpretation

The five primary series without the observed malformed-reference pattern have mean phi **0.5092--0.5768**, LR **55.00--76.67%**, and ROUGE-1 precision **0.2283--0.3284**. Their order by mean is Qwen2.5-1.5B (0.5768), Qwen2.5-3B (0.5571), Phi-1.5 (0.5420), Llama-3.2-3B (0.5365), and Gemma-2-2B (0.5092). Qwen2.5-1.5B has the highest mean, but Qwen2.5-3B has the highest LR: 76.67% versus 76.00%. Avoid identifying both maxima with the same model.

The archived-reference diagnostic means are 0.0832 for TinyLlama and 0.1297 for Phi-3.5-mini; neither has a phi > 0.5 case. Direct checks of the NEW reconstruction records confirm that 299/300 TinyLlama references and 162/300 Phi references contain no ASCII space. All 300 references in each of the five other primary series, and all 600 verified full-sequence references, contain ASCII spaces. These are format indicators, not a comprehensive semantic validation.

All **62,757** available main-cohort heuristic segments were attempted. There are **62,755 nonempty selected outputs**: Llama idx 259 ends with an empty 33rd selected segment, and Phi-1.5 idx 121 ends with an empty 17th selected segment. Neither empty output omits a subsequent segment. There are **1014** truncated encoder calls. Say “all segments were attempted, with two empty terminal outputs”; do not say every segment was successfully reconstructed or that no empty output occurred. The verified-reference TinyLlama and Phi cohorts attempt all **8835** and **10,748** segments, with **24** and **35** truncated encoder calls and no empty output.

Suggested Results opening:

> The completed campaign contains 2100 reconstructions using archived-reference traces, 600 reconstructions using separately verified full-sequence references, 240 trace-control reconstructions, 360 defense-pilot reconstructions, and 3300 MAP reconstructions. Every planned record is present and independently checked. The primary and verified-reference cohorts attempt all available heuristic segments; two primary reconstructions have an empty terminal segment. The 96-token pilots allow up to three segments, whereas the 512-token MAP experiment reconstructs one. These protocol differences prevent attributing cross-experiment score differences to victim budget alone.

## Pairwise comparisons and EOS subsets

**18 of 21** Holm-adjusted model comparisons are significant at BOTH prompt and scenario levels. The only inconclusive pairs are Qwen2.5-3B versus Llama, Qwen2.5-3B versus Phi-1.5, and Llama versus Phi-1.5. Diagnostic-series comparisons are included in that family and should not be interpreted as transfer differences on ordinarily rendered text.

Qwen2.5-1.5B minus Qwen2.5-3B: mean difference **0.0197**, prompt-bootstrap CI **[0.0055, 0.0332]**, rank-biserial **0.1962**, Holm p **0.0129**; scenario CI **[0.0059, 0.0335]**, rank-biserial **0.4011**, Holm p **0.0279**. All seven scenario mean intervals are wider than the prompt intervals; maximum widening is **44.8%** for Qwen2.5-3B. This width comparison concerns two analyses of the SAME current data, so it is permissible.

EOS-only example: Qwen2.5-3B mean increases from **0.5571** in all 300 responses to **0.6043** among its 50 EOS-complete responses; Phi-1.5 decreases from **0.5420** to **0.5108** among its 144 EOS-complete responses. These are conditional subsets, not evidence that completion causes better or worse reconstruction. No Phi-3.5-mini response reaches victim EOS. Keep victim EOS distinct from the first generated reconstructor EOS used in candidate scoring.

The descriptive six-model covariate regression has model-only R-squared **0.8624**, covariates-only **0.7213**, full **0.8699**, incremental **0.0076**. It includes both malformed-reference diagnostics and excludes the Phi-1.5 base-model series; retain its noncausal interpretation if retained at all.

## Separate verified-reference cohorts and matched rendering contrast

TinyLlama: mean **0.4677**, scenario CI **[0.4501, 0.4854]**, LR **44.33%**. Phi-3.5-mini: mean **0.5269**, scenario CI **[0.5127, 0.5411]**, LR **62.33%**. Keep these separately reported; do not insert them into the seven-model archived-reference ranking.

Matched full-sequence-minus-independent-token rendering contrasts, with BOTH reconstruction arms drawn from the CURRENT EOS campaign: TinyLlama **0.3845**, paired-scenario CI **[0.3662, 0.4029]**; Phi **0.3972**, CI **[0.3806, 0.4134]**. These contrast a change in BOTH the trace and the reference; they do not separately identify the contributions of input restoration and reference restoration. “Newly generated responses” should be replaced with “verified-reference responses” when describing this campaign: the campaign reran reconstructions from frozen responses, not victim generation.

Suggested paragraph:

> Under matched token sequences, reconstruction settings, and seeds, full-sequence rendering changes both the observed length vector and the scoring reference. The corresponding mean-score differences are 0.3845 for TinyLlama and 0.3972 for Phi-3.5-mini, with paired-scenario 95% intervals [0.3662, 0.4029] and [0.3806, 0.4134]. Both intervals exclude zero. This establishes sensitivity to the joint rendering protocol; it does not isolate which component accounts for the difference.

## Controls, defenses, and MAP

Controls: **4/6** significant Holm-adjusted contrasts. Shuffling and constant traces significantly lower the mean in BOTH models. Cross-prompt means are lower in both models but neither contrast is conclusive: Holm p **0.1169** for each. Qwen true/shuffle/constant/cross means **0.3235 / 0.1840 / 0.1169 / 0.2745**; Llama **0.2917 / 0.1870 / 0.1252 / 0.2320**. The previous claim that Llama shuffling was inconclusive is false for this dataset. These controls are partial ablations and can change segmentation/coverage; retain the limitation against a general advantage over unrelated traces.

All ten defense-versus-true comparisons are significant, each Holm p **0.0000183**. Relative mean-score reductions span **43.8--89.6%**, and batching alone spans **70.9--79.3%**. Payload overhead is measured on byte payloads, not TLS/HTTP transport cost; packet privacy, latency, traffic grouping, and alternative attackers remain unmeasured.

An independent read of NEW secondary records confirms **660 pad_32 records** (60 defense-pilot plus 600 MAP), with a single retained 80-token input within EACH model/attacker regime. Across all regimes there are TWO distinct input-ID sequences, reflecting frozen `_32` versus MAP-inverted `_3` inputs to the same FirstSentences encoder; do not claim all 660 token-ID sequences are globally identical. For frozen MAP, matched prompt/seed keys give **150/150 identical predictions between the two victim-model datasets**. This directly supports the explanation that nonzero overlap can occur without response-specific length variation in the retained input. It does not establish secrecy against other observations or attackers.

MAP has 22 current cells, 150 rows each. Mean phi ranges **0.0422--0.2852**, between-seed SD **0.0032--0.0184**. Baseline frozen Qwen/Llama means **0.1988/0.1860**; pad_32 frozen **0.2371/0.2270**, adaptive **0.1583/0.1492**. Retain exact current confidence intervals from the new secondary audit. Remove all claims that these predictions or means “match historical results”; single-candidate MAP has no within-call candidate ranking, which suffices to describe its protocol.

## Current candidate selector and complete audit

Suggested Methods replacement:

> For each sampled candidate, the selector sums the recorded token log probabilities through the first generated EOS token, inclusive; tokens after that EOS are excluded. If EOS is not generated, every generated token is included. The scores use the sampling-processed distributions under the specified top-k, temperature, and repetition constraints, with no length normalization. A nonfinite contribution within the scored prefix fails validation. Segment deduplication and subsequent context use the selected text. Single-candidate MAP calls have no confidence-based choice among alternatives.

The complete current audit checks **6600 records, 86,788 generation calls, and 253,764 candidates**: **83,488 ranked calls** and **3300 single-candidate calls**, with **4,540,261 scored token positions**. In **142,629 candidates**, **393,353 token positions after the first EOS** are excluded. Nonfinite values in excluded tails do not enter a candidate score. These counts concern the CURRENT implementation. Remove all earlier-selector changed-selection, changed-response, or “reproduced earlier MAP” counts from the paper.

Replace the old 24-case appendix with a description of complete current integrity checks: record counts and unique keys; frozen input/reference bindings; candidate-score alignment and EOS boundaries; finite scored prefixes; selected-candidate and segment-context consistency; independent lexical rescoring; and bootstrap/paired-test recomputation. Scores conditioned on this reconstructor, prompt corpus, heuristic segmentation, and byte-observation model remain limited evidence of lexical reconstruction, not semantic recovery or a demonstrated transport exploit.

## Stale-content inventory in the pre-edit 670-line manuscript

Line references identify `before/article1/template.tex` in this revision folder, not an edited live file.

- Abstract line 38: replace every result and conclusion based on old means, old 3/6 control count, old defense range, and original-selector caveat.
- Introduction lines 47, 49, 58: remove historical collector-result narrative; describe frozen verified-reference cohorts accurately.
- Measurement lines 126--142: remove old 2073/2100 and 764 collector comparison, old execution-score comparison, and bounded 24-case selector audit; specify current EOS-inclusive rule. Keep only freshly checked current reference-format counts and independently replayed rendering provenance.
- Pipeline figure line 158 and caption 170: remove “PAD-inclusive score sum” and original-selector description.
- Repeated-seed Methods around 279--287: remove historical-prediction comparisons; describe fixed registered seeds and current candidate count.
- Results opening line 304: all-segment coverage must distinguish attempted segments from nonempty outputs and must not claim victim regeneration in this campaign.
- Trace-controls paragraph and table: replace all means and significance decisions.
- Primary Results around 334--380: replace means, LR/R1 ranges, 1475 truncation count, and diagnostic means. Delete the entire paragraph beginning “Relative to the earlier stored reconstructions”.
- EOS Results: replace conditional means and remove “rates are unchanged” comparison with the earlier campaign.
- Pairwise statistics: replace 15/21, six inconclusive top-four pairs, old examples and 41.6% CI widening.
- Fresh-reference Results: replace means, LR, 55/77 truncation counts, paired score differences and intervals.
- Defense Results and discussion: replace 36.7--88.7% and old batching range; validate all current table cells.
- MAP Results: source values from current audit and remove earlier-prediction identity assertion.
- Discussion around 555--556: replace 3/6 and Llama-shuffle exception; keep cross-prompt limitation.
- Practical interpretation line 564: highest-mean Qwen1.5 has LR 76.00% and R1 precision 0.3284; highest LR belongs to Qwen3.
- Limitations line 569: remove obsolete selector-error paragraph and proposed uncompleted rerun; retain current conditional-estimand, sum-score length preference, and reproducibility limitations.
- Conclusions lines 581--585: rewrite all results and claims solely from current evidence.
- Appendix A / Data Availability: point numerical authority and current supplement inventory to EOS campaign. The earlier archive may be named only as frozen input provenance.
- Appendix C lines 652--657: replace old 24/18/38 audit with complete current candidate audit; no old-selector comparison rows.

Cited prior research remains legitimate literature context. None of the above requires erasing cited comparison figures that are explicitly attributed to other studies, or deleting immutable old experimental archives.
