# MDPI SuSy fields for this manuscript

Draft field list for the intended *Electronics* submission at https://susy.mdpi.com. The target below was confirmed by the author. No submission has been made through this file. Current Special Issue availability, deadline, and journal requirements still require verification; the completed EOS-bounded campaign, interpretation limits and package actions are listed in the submission checklist.

## Journal and article type

| Field | Value |
|---|---|
| Journal | Electronics |
| ISSN | 2079-9292 |
| Article type | Article |
| Section | Artificial Intelligence |
| Special Issue | Trustworthy AI for Large Models and Security Systems |
| SI deadline | Not currently verified |
| SI listing | https://www.mdpi.com/journal/electronics/sections/Artificial_Intell |

In SuSy, select the Special Issue from the dropdown (do not submit as a regular-section paper). If the issue does not appear, search by the exact title above or contact electronics@mdpi.com before submitting as a regular article.

## Manuscript metadata

| Field | Value |
|---|---|
| Title | Lexical Reconstruction from Instrumented LLM Token-Length Traces: Reference Validation and Defense Evaluation |
| Article type (LaTeX) | `\documentclass[electronics,article,submit,pdftex,moreauthors]{Definitions/mdpi}` |
| Cover letter | `article1/COVER_LETTER.md` (export to PDF or paste into the cover-letter box) |
| Manuscript files | `template.tex`, `references.bib`, `Definitions/`, compiled `template.pdf` |
| Data availability | Local Supplementary Data S2 `LLeakM-S2-EOS-20261102-v1`, with manifest and verifier; public archival pending. Project: https://github.com/jackaugustyn/LLeakM |
| Supplementary file | `article1/supplement/LLeakM-S2-EOS-20261102-v1.tar.gz` and its SHA-256 file |

The current evidence is in `experiment_validation/selector_eos_20261101/`; manuscript tables, figures and validation are in its `article_revision_20261102/` directory. The 600 verified-reference victim responses and token IDs were collected before this reconstruction campaign.

## Corresponding authors

- Paweł Augustynowicz, pawel.augustynowicz@wat.edu.pl
- Szymon Jędrzejczak, szymon.jedrzejczak@wat.edu.pl
- Affiliation: Faculty of Cybernetics, Military University of Technology, gen. Sylwestra Kaliskiego 2, 00-908 Warsaw, Poland
- ORCID: none (leave blank)

## One-sentence cover-letter summary for the SuSy box (if a short field is required)

Article for the Special Issue “Trustworthy AI for Large Models and Security Systems”: 6600 completed reconstructions with an EOS-bounded selector on frozen application-layer length traces, including separately verified reference cohorts and measured framing transformations, with explicit transport and encoder-coverage limits.
