# Electronics submission checklist

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
- [x] Re-run every analysis from a clean environment and verify all table and figure values against the released artifacts. See `experiment_validation/analysis/manuscript_number_verification.json`.
- [x] Preserve package versions, model revisions/commit hashes, tokenizer revisions, hardware details, and random seeds. See `experiment_validation/analysis/REPRODUCIBILITY.md` and `experiment_validation/analysis/env_snapshot.json`.
- [x] Implement an EOS-aware, longer-output collection path (`max_new_tokens=512`, `max_sentences=0`) and record `finish_reason`/`response_complete`.
- [x] Execute the longer-output matrix and include only EOS-complete responses in the complete-answer analysis; do not submit a complete-response claim before this is done. Campaign `full_20260920` is complete (7/7, 300/300). Table `tab:eos_completeness` uses `response_complete` only; Phi-3.5-mini has $n_{\mathrm{EOS}}=0$ at 512 tokens.
- [x] Implement repeated reconstruction with five decoder seeds, common random numbers across conditions, and a prompt-disjoint defense-aware MAP attacker (`scripts/robustness_eval.py`).
- [x] Execute the complete robustness matrix and update the manuscript only after `analysis/robustness_eval.json` reports `complete: true`. Protocol `028a07f6eb2ebdd2`, 22/22 cells, 3300 reconstructions. Table `tab:robustness`.
- [ ] Add at least one end-to-end encrypted-traffic capture; this remains separate from the application-event length-oracle experiment.
- [ ] Consider human evaluation or one fixed paraphrase-aware metric to complement lexical overlap.

## Manuscript checks

- [x] Confirm every model name, parameter count, license, and cited model-report version. See Table `tab:model_metadata` in `template.tex` and `experiment_validation/analysis/tables_metadata.tex`.
- [x] Verify the stated truncation range directly against the final analysis table. Per-model truncation from `samples.jsonl` `token_count` $\geq 512$ is 44.3, 83.3, 34.3, 74.3, 52.0, 41.0, 100.0\% (range 34.3--100\%). See Table `tab:length_covariates`.
- [x] Add model/tokenizer and response-length covariates to the manuscript if space permits. Added Tables `tab:model_metadata` and `tab:length_covariates`, plus the OLS covariate sentence in the statistical-results section.
- [x] Review the distinction between application-event lengths and observable TLS/HTTP/2/QUIC records throughout the paper. Abstract, introduction, threat model, figure caption, discussion, and conclusions now state that the traces are laboratory SSE application events, not TLS/HTTP/2/QUIC records.
- [x] Confirm that “SSE-equivalent” in the title accurately reflects the final experimental setup. The title refers to one-token-per-SSE-event application traces; the abstract defines the term and excludes encrypted-transport extraction.
- [ ] Have both authors proofread the English and approve all claims, figures, tables, and supplementary files. English was revised in `template.tex`; Szymon Jędrzejczak still needs to read the compiled PDF and approve.
- [x] Keep appendix execution identifiers because they are the public artifact directory names required for external reproduction (`tab:run_ids`). They were not removed.

## Electronics submission package

- [x] Select the most relevant Electronics section or Special Issue and check its scope and deadline. Section: Artificial Intelligence. Special Issue: Trustworthy AI for Large Models and Security Systems (deadline 15 November 2026). See `SUBMISSION_SYSTEM.md`.
- [x] Insert the final article type and any Special Issue information in the submission system. Article type is `Article` in `template.tex` (`electronics,article,submit`). SuSy copy-paste fields are in `SUBMISSION_SYSTEM.md`. Authors still need to select these values after logging in to https://susy.mdpi.com.
- [x] Prepare a concise cover letter stating novelty, fit with Electronics, and the controlled nature of the threat model. See `COVER_LETTER.md`.
- [ ] Supply suggested and opposed reviewers only where justified and disclose potential conflicts.
- [ ] Complete the MDPI ethics, data-availability, funding, conflicts-of-interest, and AI-use declarations in the submission system.
- [ ] Check current APC, institutional discounts, and funding approval before submission.
- [ ] Upload `template.tex`, `references.bib`, the complete `Definitions/` directory, figures, tables, supplementary files, and the compiled PDF.
- [ ] Compile the exact uploaded archive in a clean environment and inspect the final PDF page by page.
