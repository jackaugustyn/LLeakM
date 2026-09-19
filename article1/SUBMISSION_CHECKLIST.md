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
- [ ] Check that released prompts and model outputs contain no personal, confidential, copyrighted, or unsafe material requiring redaction.
- [ ] Re-run every analysis from a clean environment and verify all table and figure values against the released artifacts.
- [ ] Preserve package versions, model revisions/commit hashes, tokenizer revisions, hardware details, and random seeds.
- [ ] Consider adding longer-output experiments because the 96-token budget truncates most responses.
- [ ] Consider evaluating an adaptive attacker trained on defended traces and at least one end-to-end encrypted-traffic capture.
- [ ] Consider human evaluation or one fixed paraphrase-aware metric to complement lexical overlap.

## Manuscript checks

- [ ] Confirm every model name, parameter count, license, and cited model-report version.
- [ ] Verify the stated 95--100% truncation range directly against the final analysis table.
- [ ] Add model/tokenizer and response-length covariates to the manuscript if space permits.
- [ ] Review the distinction between application-event lengths and observable TLS/HTTP/2/QUIC records throughout the paper.
- [ ] Confirm that “SSE-equivalent” in the title accurately reflects the final experimental setup.
- [ ] Have both authors proofread the English and approve all claims, figures, tables, and supplementary files.
- [ ] Remove execution identifiers from the appendix if they do not help external reproduction.

## Electronics submission package

- [ ] Select the most relevant Electronics section or Special Issue and check its scope and deadline.
- [ ] Insert the final article type and any Special Issue information in the submission system.
- [ ] Prepare a concise cover letter stating novelty, fit with Electronics, and the controlled nature of the threat model.
- [ ] Supply suggested and opposed reviewers only where justified and disclose potential conflicts.
- [ ] Complete the MDPI ethics, data-availability, funding, conflicts-of-interest, and AI-use declarations in the submission system.
- [ ] Check current APC, institutional discounts, and funding approval before submission.
- [ ] Upload `template.tex`, `references.bib`, the complete `Definitions/` directory, figures, tables, supplementary files, and the compiled PDF.
- [ ] Compile the exact uploaded archive in a clean environment and inspect the final PDF page by page.
