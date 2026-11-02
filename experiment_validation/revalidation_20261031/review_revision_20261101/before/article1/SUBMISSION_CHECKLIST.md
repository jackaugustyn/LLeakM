# Electronics submission checklist

Revision status (1 November 2026): all 12 computation/verification stages completed, comprising 6600 T5 reconstructions and 600 fresh victim generations. The manuscript and 21-page PDF are aligned with these verified results. All 12 tables, 476 numerical fields and all four revised figures passed the numerical, scientific or visual checks appropriate to their content; reproducible plot sources and audit records are in `article_revision_latest_20261101/`. Archived malformed references remain diagnostic; new reference cohorts and their checkpoint provenance are reported separately. Submission actions below remain distinct from local completion.

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
- [x] Reconstruct all 2100 archived responses from exact verified logged payload lengths; all new encoder inputs and code/checkpoint hashes are retained. The unavailable historical collector inputs are explicitly distinguished from this completed rerun. See `revalidation_20261031/inference_validation/cuda_full/verification.json` (15/21 significant pairs in the new results).
- [x] Generate 300 new responses for each of TinyLlama and Phi-3.5-mini, record token IDs and validate full-sequence decoding, then reconstruct and verify all 600. Keep these cohorts separate from the diagnostic archived series; checkpoint/environment differences prevent a pure detokenization-effect estimate.
- [x] Record current code, checkpoints, environment, transformation seeds and exact reconstruction inputs. Preserve the unresolved historical provenance as a limitation; `analysis/repro_manifest.json` and `analysis/revision_snapshot_20261031.json` are earlier snapshots, not attestations of the rerun. The completion manifest predates final manuscript integration.
- [x] Rerun and validate all 240 control and 360 defense trials. Retain actual random-padding vectors and realized costs. Three of six control contrasts and ten of ten defense contrasts are significant after the respective Holm corrections.
- [ ] Include the newly generated revalidation artifacts and final manuscript files in the public submission release; a local completion manifest does not establish public availability.
- [ ] Review the new model outputs and their release contents; the earlier release audit above covers the previous archive only.
- [x] Implement an EOS-aware, longer-output collection path (`max_new_tokens=512`, `max_sentences=0`) and record `finish_reason`/`response_complete`.
- [x] Execute the longer-output matrix and include only EOS-complete responses in the complete-answer analysis; do not submit a complete-response claim before this is done. Campaign `full_20260920` is complete (7/7, 300/300). Table `tab:eos_completeness` uses `response_complete` only; Phi-3.5-mini has $n_{\mathrm{EOS}}=0$ at 512 tokens.
- [x] Implement repeated reconstruction with five decoder seeds, common random numbers across conditions, and a prompt-disjoint defense-aware MAP attacker (`scripts/robustness_eval.py`).
- [x] Rerun all 22 MAP settings and 3300 reconstructions, retaining exact clean/transformed/inverse lengths and encoder tokens. `revalidation_20261031/secondary/cuda_full/summary.json` reports the complete new design. The protocol uses 512-token-budget references, the first heuristic segment, one candidate per seed, five seeds and 80-token input/output caps; it is not an all-segment robustness run. Historical outputs remain in `analysis/robustness_eval.json`.
- [ ] Add at least one end-to-end encrypted-traffic capture; this remains separate from the application-event length-oracle experiment.
- [ ] Consider human evaluation or one fixed paraphrase-aware metric to complement lexical overlap.

## Manuscript checks

- [x] Confirm every model name, parameter count, license, and cited model-report version. See Table `tab:model_metadata` in `template.tex` and `experiment_validation/analysis/tables_metadata.tex`.
- [x] Verify the stated truncation range directly against the final analysis table. Per-model truncation from `samples.jsonl` `token_count` $\geq 512$ is 44.3, 83.3, 34.3, 74.3, 52.0, 41.0, 100.0\% (range 34.3--100\%). See Table `tab:length_covariates`.
- [x] Add model/tokenizer and response-length covariates to the manuscript if space permits. Added Tables `tab:model_metadata` and `tab:length_covariates`, with historical response covariates distinguished from new reconstruction metrics; any OLS sentence must name its actual data source.
- [x] Review the distinction between application-event lengths and observable TLS/HTTP/2/QUIC records throughout the paper. Abstract, introduction, threat model, figure caption, discussion, and conclusions now state that the traces are laboratory SSE application events, not TLS/HTTP/2/QUIC records.
- [x] Align the manuscript and submission-document title: “Information Leakage from Token-Length Traces in Streaming LLMs: Cross-Model Reconstruction and Defense Evaluation”. The revision distinguishes verified logged payload lengths, archived collector defects and fresh reference cohorts; it does not claim encrypted-transport extraction.
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
