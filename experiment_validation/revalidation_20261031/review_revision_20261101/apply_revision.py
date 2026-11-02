"""Replay source-bound, narrowly targeted review edits from the preserved draft.

This is a local exact-text patch record, not an ARS schema-670 authorization
certificate. The user's instruction adopts the nonsprint review's changes.
"""
from pathlib import Path
import hashlib
import json
import re
import difflib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE / 'before/article1/template.tex'
text = BASE.read_text()
ops = []


def edit(ids, old, new):
    global text
    assert text.count(old) == 1, (ids, old[:100], text.count(old))
    ops.append({'findings': ids.split(), 'old': old, 'new': new})
    text = text.replace(old, new, 1)


def para(ids, start, new):
    old = next(p for p in text.splitlines() if p.startswith(start))
    edit(ids, old, new)


OLD_TITLE = 'Information Leakage from Token-Length Traces in Streaming LLMs: Cross-Model Reconstruction and Defense Evaluation'
TITLE = 'Lexical Reconstruction from Instrumented LLM Token-Length Traces: Reference Validation and Defense Evaluation'
edit('EIC-W1', OLD_TITLE, TITLE)
edit('R1-W1', 'We evaluate a fixed T5 reconstructor through 6600 reconstructions',
     'We evaluate a fixed T5 implementation with its recorded candidate-selection rule through 6600 reconstructions')
edit('R1-W2', 'a maximum-a-posteriori (MAP) length inverse',
     'a local empirical maximum-a-posteriori (MAP) length inverse')

sources = (HERE / 'sources_proposal.md').read_text()
source_paras = re.findall(r'```latex\n(.*?)\n```', sources, re.S)
edit('R2-W2', r'For generative AI, Weiss \emph{et al.} introduced T5-based sequence-to-text inversion and reported accurate reconstruction for 27\% of responses and topic inference for 53\% in their evaluated setting~\cite{weiss2024prompt}.', source_paras[0])

edit('R1-W1', r'\subsection{Reconstruction Pipeline}', r'\subsection{Reconstruction Pipeline}\label{sec:reconstructor}')
edit('R1-W1', 'Candidates are ranked by the transition log-probability sum at each segment, and the chosen text supplies the next segment\'s context.',
     'The executed selector sums normalized transition scores over all returned generation steps, including PAD positions after a candidate\'s first end-of-sequence (EOS) token; the chosen text supplies the next segment\'s context.')
anchor = 'The executed selector sums normalized transition scores over all returned generation steps, including PAD positions after a candidate\'s first end-of-sequence (EOS) token; the chosen text supplies the next segment\'s context.'
edit('R1-W1 R2-W3', anchor, anchor + '\n\n' + r'''These scores are obtained after the sampling constraints, including top-$k$ and the no-repeat rule; their sum is not an unconstrained model likelihood. The implementation does not mask post-EOS steps, which can assign nonfinite scores to a completed candidate and alter selection. A bounded replay reproduced all 24 selected first-segment predictions, but ranking only through the first EOS changed the selected text in 18 cases (Appendix~\ref{app:selector}). All three-candidate results in this paper retain the original PAD-inclusive selector. They describe this specific implementation and do not estimate the performance of an EOS-masked selector. The one-candidate MAP experiment bypasses candidate ranking.

''' + source_paras[1])

edit('R2-W1 DA-N1', 'The lexical score $\\phi$ does not estimate mutual information, and its reduction for a fixed attacker does not measure defended-channel capacity.', r'''The lexical score $\phi$ does not estimate mutual information, and its reduction for a fixed attacker does not measure defended-channel capacity.

For a single-segment reconstruction, let $E(Z)$ denote the deterministic encoder input after inversion (if used), segmentation, serialization and truncation. Then the chain refines to $R\rightarrow X\rightarrow Z\rightarrow E(Z)\rightarrow\widehat R$, and $I(R;E(Z))\leq I(R;Z)$. The complete trace $Z$ may retain variation that never reaches the decoder. This distinction matters for \texttt{pad\_32}: the audited single-call encoder inputs are constant within each evaluated condition (Section~\ref{sec:defense}). Multi-segment decoding additionally uses the previously selected text as context, as specified below.''')
edit('R1-W1 R2-W1', r'{Segment + context\\serialize and tokenize\\encoder limit: 80 tokens}', r'{Retained encoder input\\segment + context\\limit: 80 tokens}')
edit('R1-W1', r'{Rank candidates\\select one text per segment}', r'{PAD-inclusive score sum\\select one text per segment}')
edit('R1-W1 EIC-W2', 'Candidate selection precedes each context update; the first segment has no reconstructed context.', 'The original PAD-inclusive selector precedes each context update (Section~\\ref{sec:reconstructor}); the first segment has no reconstructed context.')

methods = (HERE / 'methods_proposal.tex').read_text()
table = methods[methods.index(r'\begin{table}'):methods.index(r'\end{table}') + len(r'\end{table}')]
# Keep one protocol table compact; the grouping column identifies the exact design.
table = table.replace('including two diagnostics', '300 per model; two diagnostic series')
table = table.replace('TinyLlama and Phi-3.5-mini\n& 600', '300 per model, TinyLlama and Phi-3.5-mini\n& 600')
table = table.replace('Archived text; Qwen2.5-1.5B and Llama-3.2-3B', 'Archived text; two models, $30\\times4$ per model')
table = table.replace('Archived text; same two models\n& 360', 'Archived text; two models, $30\\times6$ per model\n& 360')
table = table.replace('Archived text; same two models\n& 3300', 'Archived text; two models, $30\\times5\\times11$ per model\n& 3300')
table = table.replace('The last column refers', 'The controls, defense pilot and MAP study use Qwen2.5-1.5B and Llama-3.2-3B. Samples are candidates per segment, not independent victim responses. The last column refers')
edit('EIC-W2', r'\subsection{Prompt Corpus and Experimental Protocol}', r'\subsection{Prompt Corpus and Experimental Protocol}' + '\nTable~\\ref{tab:protocol_overview} summarizes the cohorts, reconstruction scopes and seed rules.\n\n' + table)

stat = methods.split('% R1-W3:')[1].split('% Keep the existing')[0].split('\n', 1)[1].strip()
oldstat = ('Because the same 300 prompts were issued to every model, comparisons of the retained scores used prompt-aligned observations. '
           r'Mean metrics were accompanied by percentile-bootstrap 95\% confidence intervals~\cite{efron1994bootstrap} based on $B=10{,}000$ prompt-index resamples with replacement and seed \texttt{20260706}.')
edit('R1-W3', oldstat, stat)
edit('R1-W3', 'resamples those scenario means 10,000 times with the same seed', "resamples those scenario means 10,000 times from the same aggregator's random-number stream")
maptext = methods.split('% R1-W2:')[1].split('% Continue')[0].split('\n', 1)[1].strip()
oldmap = ('The adaptive attacker is an exact-prompt-disjoint empirical MAP inverse of each defense, followed by the unchanged T5 decoder. '
          'It learns the empirical distribution of logged token lengths (or, for batching, of length tuples) conditional on the defended observation from the 270 calibration traces of the same victim model. Test responses are never used for fitting.')
edit('R1-W2', oldmap, maptext)
edit('R2-W1', 'all 30 unique inputs per model remain truncated', 'all 30 prompt-indexed inputs per model remain truncated')
edit('R2-W1', '30 unique test inputs per model). Each input is reused across five decoder seeds.', '30 test prompts per model). Each condition is repeated for five decoder seeds; identical encoder inputs can occur for different prompts.')

para('R2-W1 DA-N1', 'These results describe the complete frozen pipeline,', r'''These results describe the complete frozen pipeline, including changes in heuristic segmentation and encoder truncation after each transformation. They do not isolate information removed by a defense. In all 60 \texttt{pad\_32} pilot records, the retained encoder input is identical and reconstruction makes one decoder call. An extended audit covering these records and all 600 \texttt{pad\_32} records in the five-seed study finds one unique retained input within each suite, model and attacker group. Although the full 512-token-budget traces differ in their number of positions, that variation is removed before decoding by the 80-token encoder limit. The positive lexical scores therefore cannot demonstrate this decoder's use of response-specific trace-length variation. They reflect overlap between the references and outputs generated from the constant retained input under the recorded seeds; another attacker could still use information in the complete observable trace.''')
edit('R2-W1 DA-N1', 'This result is consistent with generic decoder outputs overlapping the references, but the comparison does not isolate that mechanism from changed segment length, encoder truncation, or retained trace length.', r'''The frozen \texttt{pad\_32} encoder input is identical across both models and all 300 reconstructions; all 150 cross-model pairs with common prompt and seed indices produce identical predictions. Their different mean scores arise from scoring those predictions against different references. Response-specific trace-length variation does not reach this decoder.''')
edit('R2-W1', 'For \\texttt{pad\\_32}, applying the MAP inverse reduced mean $\\phi$ to 0.1583', 'For \\texttt{pad\\_32}, the MAP inverse also yields a constant retained input within each model. Applying it reduced mean $\\phi$ to 0.1583')
edit('R2-W1 DA-N1', 'Constant-width traces likewise retain response length, and overlap from their generic outputs cannot be attributed exclusively to the language prior.', r'''For \texttt{pad\_32}, the complete observable trace retains its number of positions, but the actual retained encoder input is constant within each condition. Its positive lexical scores are consequently not evidence that this decoder exploits response-specific length variation. This finding concerns $E(Z)$ and does not establish absence of information in $Z$ for other attackers.''')

limitation = r'''The three-candidate experiments retain a selector that includes PAD scores after EOS. The bounded replay establishes actual selection changes under EOS masking, but does not estimate their frequency across the campaign or their effect on full-response scores. Later segments condition on the selected text, so corrected full-response results require a separate end-to-end campaign with the revised selector and retained candidates. Current score rankings, confidence intervals, rendering contrasts and defense comparisons are conditional on the original implementation. The one-candidate MAP results are unaffected by this selection mechanism.'''
edit('R1-W1', r'\subsection{Limitations}', r'\subsection{Limitations}' + '\n' + limitation + '\n')
edit('R1-W1 R2-W1', 'Practical privacy claims require evidence beyond these lexical scores.', r'''These conclusions are conditional on the recorded implementation: the three-candidate experiments use the original PAD-inclusive selector, whereas MAP uses one candidate. Constant \texttt{pad\_32} encoder inputs also show why positive lexical overlap need not demonstrate use of response-specific trace variation. Practical privacy claims require evidence beyond these lexical scores.''')

para('EIC-W3', 'The accompanying artifact package separates', r'''Supplementary Data S1 is the frozen local package \texttt{LLeakM-S1-20261101-v1}, supplied as a compressed archive with a SHA-256 manifest, a README and a CPU-only verification entry point. It contains the completed records, analysis code, source logs, model revision identifiers and manuscript files; model weights are obtained separately from the pinned checkpoints. The archive is prepared for accompanying submission and can be requested from the corresponding authors; public archival of this version remains pending. The source project repository is \url{https://github.com/jackaugustyn/LLeakM}. Table~\ref{tab:run_ids} identifies the original 512-token source directories. Results reside under \path{experiment_validation/revalidation_20261031/}: \path{inference/cuda_full/} contains the 2100 reconstructions from canonical archived lengths, \path{secondary/cuda_full/} the 3900 control, defense and MAP records, and \path{inference/fresh_tinyllama_cuda/} and \path{inference/fresh_phi_cuda/} the two 300-response new-reference reconstructions. Their generation artifacts are under \path{fresh_references/}.''')
edit('EIC-W3 R1-W1', 'The earlier \\path{analysis/repro_manifest.json}', r'''The \path{review_revision_20261101/} directory records the present source-bound edits, the candidate-selection audit, exact local-MAP and seed checks, and the new manuscript validation. The supplementary manifest binds the files actually distributed in S1; its verification entry point checks those hashes, inventory completeness and per-record lexical cosine without loading a model. Version-pinned reconstructor model cards and their retrieved text are included in the source-check evidence.

The earlier \path{analysis/repro_manifest.json}''')

appendix = methods[methods.index(r'\section{Local Length Inversion'):].strip()
appendix = appendix.replace('The main and verified-reference reconstructions use the exact per-record \\texttt{reconstruction\\_seed} in their immutable input inventories.', 'The immutable main and verified-reference inventories store each exact seed in the \\texttt{reconstruction\\_seed} field.')
# Empty calibration is outside the defined experiment; retain its code, not prose.
appendix = appendix.replace('The code\'s empty-calibration defaults are mode 1 and fallback support $\\{0,\\ldots,17\\}$; the present experiments use nonempty, 270-trace calibration sets. ', '')
selector = r'''\section{Bounded Candidate-Selection Audit}\label{app:selector}
The completed three-candidate campaign did not retain every sampled output ID and transition score. We therefore replayed 24 first-segment inputs with the original first-segment checkpoint, Transformers 4.46.3, float32 CUDA execution and saved per-record seeds. Before inference, selection was fixed to input-length extrema for two inputs from each of the seven archived and two verified-reference cohorts, plus six defense-pilot inputs covering baseline, bucketing, fixed padding, both batching sizes and random padding. This deliberately selected diagnostic set is not a random sample of the campaign.

All 24 original selected texts were reproduced exactly. The audit retained all 72 candidate sequences and step scores, and compared the executed full-step sum with a sum through the first generated EOS, including that EOS and excluding subsequent PAD. For a sequence without EOS, all generated steps are retained. The two rules selected different text in 18 of 24 cases: 11 of 14 archived inputs, three of four verified-reference inputs and four of six defense-pilot inputs. Post-EOS tails occurred in 38 of 72 candidates; each such tail included a nonfinite score, while no nonfinite score occurred before or at EOS. The normalization uses the sampling-processed scores, not the unconstrained model logits.

These findings establish that the defect can affect actual generated candidates. They do not estimate its prevalence over the campaign, a corrected full-response mean or the direction of change in any table. The original selector remains the definition of the three-candidate results reported here. Revising that definition requires replay of later segments because their context depends on the selected text. The one-candidate MAP study does not use this ranking. The supplementary \path{review_revision_20261101/ranking_audit/} directory contains the fixed input selection, complete candidate records, EOS-masking tests, protocol and audit report.
'''
edit('R1-W1 R1-W2 R1-W3', r'\authorcontributions{', appendix + '\n\n' + selector + '\n\n' + r'\authorcontributions{')
para('EIC-W3', r'\dataavailability{', r'''\dataavailability{The source project repository is \url{https://github.com/jackaugustyn/LLeakM}. The frozen local Supplementary Data S1 package, \texttt{LLeakM-S1-20261101-v1}, contains completed reconstructions, generated token IDs, source logs, reference-validation records, analysis code and a file-level SHA-256 manifest with a verification entry point. It is prepared for accompanying submission and available from the corresponding authors on request; public archival of this completed version remains pending. Appendix~\ref{app:runs} identifies its version, contents and provenance boundaries.}''')

patch = {'schema': 'local-exact-text-review-patch/1', 'base_sha256': hashlib.sha256(BASE.read_bytes()).hexdigest(),
         'author_request': 'Ok, wprowadź zmiany z recenzji', 'review': 'reviews/ars_electronics_20261101/RECENZJA.md',
         'scope': 'Adopt all review topics; retain literal original-selector results, add bounded audit. No external publication.',
         'abstract_handling': 'Only scoped wording edits; abstract not regenerated.', 'operations': ops}
(HERE / 'revision_patch.json').write_text(json.dumps(patch, ensure_ascii=False, indent=2) + '\n')
(HERE / 'template.rev1.tex').write_text(text)
(ROOT / 'article1/template.tex').write_text(text)
(HERE / 'manuscript.diff').write_text(''.join(difflib.unified_diff(BASE.read_text().splitlines(True), text.splitlines(True), fromfile='before/template.tex', tofile='article1/template.tex')))
bib_base = (HERE / 'before/article1/references.bib').read_text()
bib_new_entry = re.search(r'```bibtex\n(.*?)\n```', sources, re.S)[1]
bib_new = re.sub(r'@inproceedings\{li2026length,.*?\n\}', lambda m: bib_new_entry, bib_base, count=1, flags=re.S)
(ROOT / 'article1/references.bib').write_text(bib_new)
(HERE / 'bibliography.diff').write_text(''.join(difflib.unified_diff(bib_base.splitlines(True), bib_new.splitlines(True), fromfile='before/references.bib', tofile='article1/references.bib')))
report = {'status': 'APPLIED', 'patch_scope': 'Local exact-text edits, not formal ARS authorization certification',
          'operations': len(ops), 'source_finding_ids': sorted({i for op in ops for i in op['findings']}),
          'base_sha256': patch['base_sha256'], 'result_sha256': hashlib.sha256(text.encode()).hexdigest(),
          'untouched_text': 'Only the uniquely matched old strings in revision_patch.json were replaced; replay is deterministic.'}
(HERE / 'apply_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
