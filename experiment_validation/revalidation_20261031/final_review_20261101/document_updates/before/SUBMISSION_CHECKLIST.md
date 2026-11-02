# Electronics submission checklist

Revision status: the review-driven text corrections do not establish that the underlying collection and reference defects are repaired. The items below distinguish recorded analyses, outstanding measurement validation, and submission actions.

## Required author decisions

- [x] Add Szymon Jędrzejczak's institutional email address.
- [x] Confirm that ORCID identifiers are not available for either author.
- [x] Confirm that Paweł Augustynowicz and Szymon Jędrzejczak are the corresponding authors and that the author order is final.
- [x] Review and approve the CRediT author-contribution statement.
- [x] Confirm that this research received no external funding and update the Funding Statement accordingly.
- [x] Confirm that the authors declare no conflict of interest and update the declaration accordingly.
- [x] Confirm that an acknowledgments section is not required.

## Research and reproducibility

- [x] Deposit code, prompts, per-sample outputs, analysis scripts, and the artifact manifest in a stable public repository (`https://github.com/jackaugustyn/LLeakM`).
- [x] Add the public repository URL to the Data Availability Statement and the main text.
- [ ] Optionally mint an archival DOI (for example, Zenodo) from a tagged GitHub release and add it to the Data Availability Statement.
- [x] Check that released prompts and model outputs contain no personal, confidential, copyrighted, or unsafe material requiring redaction. See `experiment_validation/analysis/RELEASE_CONTENT_AUDIT.md`.
- [x] Recalculate the main statistics from the archived per-sample outputs: the review reproduced them within 1e-10 and confirmed 17/21 pairwise comparisons after Holm correction. See `reviews/ars_electronics_20261031/artifacts/analysis_rerun/comparison.json`. This does not verify signal collection or a clean rerun of the full experiment.
- [ ] Establish the exact historical collector/reconstructor inputs and verify corrected reconstructions where necessary. Replay of the current collector on the archived logs changes 2073/2100 traces; it does not establish which exact inputs the historical T5 run received.
- [ ] Validate full-sequence and incremental detokenization before using TinyLlama or Phi-3.5-mini to support transfer claims on ordinary responses. The revision retains their recorded scores as apparatus diagnostics.
- [ ] Complete historical code, environment, and per-sample seed provenance. Existing `analysis/repro_manifest.json`, `analysis/REPRODUCIBILITY.md`, and `analysis/env_snapshot.json` do not prove the execution history; the manifest contains stale protocol/hash entries. `analysis/revision_snapshot_20261031.json` is a separate current-file snapshot, not a replacement historical attestation.
- [x] Implement an EOS-aware, longer-output collection path (`max_new_tokens=512`, `max_sentences=0`) and record `finish_reason`/`response_complete`.
- [x] Execute the longer-output matrix and include only EOS-complete responses in the complete-answer analysis; do not submit a complete-response claim before this is done. Campaign `full_20260920` is complete (7/7, 300/300). Table `tab:eos_completeness` uses `response_complete` only; Phi-3.5-mini has $n_{\mathrm{EOS}}=0$ at 512 tokens.
- [x] Implement repeated reconstruction with five decoder seeds, common random numbers across conditions, and a prompt-disjoint defense-aware MAP attacker (`scripts/robustness_eval.py`).
- [x] Execute the recorded robustness matrix: `analysis/robustness_eval.json` reports `complete: true`, protocol `028a07f6eb2ebdd2`, 22/22 settings and 3300 reconstructions. This run uses 512-token-budget victim traces but only the first heuristic segment, one T5 sample per seed, and the full response reference; both T5 input and output are limited to 80 tokens. It is not an all-segment robustness run. Table `tab:robustness`.
- [ ] Add at least one end-to-end encrypted-traffic capture; this remains separate from the application-event length-oracle experiment.
- [ ] Consider human evaluation or one fixed paraphrase-aware metric to complement lexical overlap.

## Manuscript checks

- [x] Confirm every model name, parameter count, license, and cited model-report version. See Table `tab:model_metadata` in `template.tex` and `experiment_validation/analysis/tables_metadata.tex`.
- [x] Verify the stated truncation range directly against the final analysis table. Per-model truncation from `samples.jsonl` `token_count` $\geq 512$ is 44.3, 83.3, 34.3, 74.3, 52.0, 41.0, 100.0\% (range 34.3--100\%). See Table `tab:length_covariates`.
- [x] Add model/tokenizer and response-length covariates to the manuscript if space permits. Added Tables `tab:model_metadata` and `tab:length_covariates`, plus the OLS covariate sentence in the statistical-results section.
- [x] Review the distinction between application-event lengths and observable TLS/HTTP/2/QUIC records throughout the paper. Abstract, introduction, threat model, figure caption, discussion, and conclusions now state that the traces are laboratory SSE application events, not TLS/HTTP/2/QUIC records.
- [x] Align the manuscript and submission-document title: “Information Leakage from Token-Length Traces in Streaming LLMs: Cross-Model Reconstruction and Defense Evaluation”. The revision distinguishes the intended application-layer observable from the current collector and does not claim encrypted-transport extraction.
- [ ] Have both authors proofread the English and approve all claims, figures, tables, and supplementary files. English was revised in `template.tex`; Szymon Jędrzejczak still needs to read the compiled PDF and approve.
- [x] Keep appendix execution identifiers because they are the public artifact directory names required for external reproduction (`tab:run_ids`). They were not removed.

## Electronics submission package

- [x] Confirm the intended target with the author: Electronics, Section Artificial Intelligence, Special Issue “Trustworthy AI for Large Models and Security Systems”. See `SUBMISSION_SYSTEM.md`.
- [ ] Verify the current official Special Issue scope, availability, deadline, and journal requirements. The review could not access the MDPI pages, so the earlier deadline entry is not treated as verified.
- [ ] Insert the article type and Special Issue in the submission system after logging in to https://susy.mdpi.com. The local LaTeX target is `electronics,article,submit`; draft SuSy fields are in `SUBMISSION_SYSTEM.md`.
- [x] Prepare a concise cover letter stating novelty, fit with Electronics, and the controlled nature of the threat model. See `COVER_LETTER.md`.
- [ ] Supply suggested and opposed reviewers only where justified and disclose potential conflicts.
- [ ] Complete the MDPI ethics, data-availability, funding, conflicts-of-interest, and AI-use declarations in the submission system.
- [ ] Check current APC, institutional discounts, and funding approval before submission.
- [ ] Upload `template.tex`, `references.bib`, the complete `Definitions/` directory, figures, tables, supplementary files, and the compiled PDF.
- [ ] Compile the exact uploaded archive in a clean environment and inspect the final PDF page by page.
