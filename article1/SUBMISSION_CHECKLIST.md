# Electronics submission checklist

Revision status (2 November 2026): the completed EOS-bounded campaign supplies all current empirical results. It contains 6600 T5 reconstructions on frozen inputs, including 600 references with previously collected and verified victim token IDs. This campaign did not generate victim responses. The article retains separate archived-reference diagnostics and verified-reference cohorts. Current evidence and manuscript integration are in `experiment_validation/selector_eos_20261101/` and its `article_revision_20261102/` directory. Local Supplementary Data S2 is `LLeakM-S2-EOS-20261102-v1`; public archival remains pending.

## Required author decisions

- [x] Add Szymon Jędrzejczak's institutional email address.
- [x] Confirm that ORCID identifiers are not available for either author.
- [x] Confirm that Paweł Augustynowicz and Szymon Jędrzejczak are the corresponding authors and that the author order is final.
- [x] Review and approve the CRediT author-contribution statement.
- [x] Confirm that this research received no external funding and update the Funding Statement accordingly.
- [x] Confirm that the authors declare no conflict of interest and update the declaration accordingly.
- [x] Confirm that an acknowledgments section is not required.

## Research and reproducibility

- [x] Complete and audit all 6600 EOS-bounded reconstructions: 2100 main, 600 verified-reference, 240 controls, 360 defense-pilot and 3300 MAP trials. See `selector_eos_20261101/validation/audit_campaign.json` and `selector_eos_20261101/completed_manifest.json` under `experiment_validation/`.
- [x] Bind the current scores to the saved inputs, references and candidate evidence. The selector includes the first EOS and excludes later score positions; input and output limits remain 80 T5 tokens.
- [x] Keep the two diagnostic reference series separate from the verified-reference results: means 0.4677 (TinyLlama) and 0.5269 (Phi-3.5-mini). The matched rendering analysis changes both input lengths and scoring references, and does not isolate those effects.
- [x] Report the current inference: 18/21 model pairs in both prompt- and scenario-level analyses; 4/6 trace-control contrasts and 10/10 defense-pilot contrasts after their respective Holm corrections. See `validation/main_fresh_factledger.json` and `validation/secondary_audit.json`.
- [x] Report realized padding costs and the first-segment, one-candidate, five-seed MAP protocol. MAP calibration is disjoint by prompt, not by scenario, and is not T5 retraining.
- [x] Regenerate all 13 tables and both numeric figures from the EOS campaign. The independent generated-table check covers 433 numeric fields and 93 descriptive checks; source bindings and figure data are retained in `article_revision_20261102/`.
- [x] Identify the project repository (`https://github.com/jackaugustyn/LLeakM`) without implying that the current local package has been publicly deposited.
- [ ] Include the final manuscript and Supplementary Data S2 in the public submission release; a local manifest does not establish public availability.
- [ ] Review the current package contents and model outputs for release. The earlier `analysis/RELEASE_CONTENT_AUDIT.md` covers its historical archive only.
- [ ] Optionally mint an archival DOI from a tagged public release and add it to the Data Availability Statement.

## Manuscript and submission checks

- [x] Compile and visually inspect all 24 pages of the final local PDF. The complete manuscript check confirms 486 table/plot values and 121 descriptive or artifact-binding checks; the independent scientific review has no unresolved findings. See `article_revision_20261102/final_validation.json`.

- [x] Preserve the approved title, author information and intended target: Electronics, Section Artificial Intelligence, Special Issue “Trustworthy AI for Large Models and Security Systems”. Draft SuSy fields are in `SUBMISSION_SYSTEM.md`.
- [x] State the instrumented application-layer threat model and distinguish lexical scores from semantic disclosure and encrypted-transport extraction.
- [x] Retain model/tokenizer metadata, source-response identifiers and the distinction between victim EOS termination and reconstruction quality. Metadata consistency is a local check, not current model-card or license certification.
- [x] Update the cover letter and data-availability package identifier to the current EOS campaign and Supplementary Data S2.
- [ ] Have both authors approve the final claims, tables, figures, English and supplementary files. Szymon Jędrzejczak still needs to read and approve the compiled PDF.
- [ ] Verify current official Special Issue availability, deadline and journal requirements; no current official verification is claimed.
- [ ] Enter the confirmed target and declarations in SuSy after logging in at https://susy.mdpi.com. No submission has been made.
- [ ] Supply reviewers and disclose reviewer conflicts if required by the submission form.
- [ ] Complete the ethics, data-availability, funding, conflicts-of-interest and AI-use declarations in the submission system.
- [ ] Check current APC, institutional discounts and funding approval before submission.
- [ ] Upload the final TeX, bibliography, `Definitions/`, figures, compiled PDF and `article1/supplement/LLeakM-S2-EOS-20261102-v1.tar.gz` with its SHA-256 file.
- [ ] Compile the exact uploaded archive in a clean environment and inspect the final PDF page by page.
