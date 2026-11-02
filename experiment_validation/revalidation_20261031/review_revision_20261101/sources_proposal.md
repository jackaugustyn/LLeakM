# Source-bound revision proposal

Scope: R2-W2, R2-W3, and the auxiliary correction to `li2026length`. No changes made to the shared manuscript or bibliography. Primary-source hashes and complete pinned model-card texts are retained in `sources_proposal.json`.

## R2-W2: replace the opening sentence of Related Work paragraph at original line 65

```latex
For generative AI, Weiss \emph{et al.} introduced T5-based sequence-to-text inversion and, in their first-segment evaluation, reported 27\% of reconstructions above an embedding-cosine threshold of 0.9 and approximately 53\% above 0.5~\cite{weiss2024prompt}. Their cosine uses MiniLM sentence embeddings; its thresholds and first-segment scope are not directly comparable to our term-frequency cosine and full-response $\mathrm{LR}@0.5$.
```

Leave the remainder beginning “Li \emph{et al.} subsequently…” unchanged. Evidence: Weiss printed p. 3375, Sections 5.1–5.2, cached text lines 621–642. The exact lower-threshold percentage is reported as **over 52.7%**, rounded in the abstract to 53%. The inequalities are strict, as reflected by “above”. The manuscript's reference to accurate *responses* omitted the first-segment scope.

## R2-W3: add after naming the first/middle checkpoints in Reconstruction Pipeline

This paragraph can follow the existing paragraph, avoiding a second occurrence of checkpoint names:

```latex
The pinned model cards identify these checkpoints as the T5-Large fine-tunes used by Weiss \emph{et al.}, trained on the first answers from UltraChat's ``Questions About the World'' subset. The source procedure trains separate models for first segments and subsequent segments, with preceding-segment text providing context for the latter~\cite{weiss2024prompt}. We verify the downloaded checkpoint revisions but do not independently reproduce training or audit its underlying records. The original paper defines token lengths in characters; our traces measure UTF-8 bytes of emitted fragments. These units agree for ASCII text but can differ for non-ASCII text, adding a representation shift alongside differences in tokenizer and response style.
```

The two model-card URLs below provide direct, version-pinned provenance. They can be linked in the existing artifact appendix, or included in the supplementary manifest instead of adding two long footnotes:

- `https://huggingface.co/royweiss1/T5_FirstSentences/blob/057cf090b719f5cf89da504babd2122d404e0953/README.md`
- `https://huggingface.co/royweiss1/T5_MiddleSentences/blob/07e2b3fc96bc129eb4a2871e58731c07e889cecc/README.md`

Both raw endpoints returned HTTP 200 on 2026-11-01. Both explicitly say “This is the model used in the USENIX Security 24' paper,” identify T5-Large, and describe UltraChat “Questions About the world” and first answers. Thus there is a documented link; saying that the training provenance is wholly undocumented would be too strong. The independent-verification caveat remains necessary.

Evidence from Weiss: printed pp. 3373–3374, Sections 4.5–5.1, cached text lines 531–589; character definition at printed p. 3370, Section 3.1, cached text lines 260–273. The paper describes first-answer selection and separate first/middle segment training. There is no need to add source epoch counts to the manuscript to resolve this concern.

## Auxiliary bibliography correction

Replace the current `li2026length` entry with:

```bibtex
@inproceedings{li2026length,
  author    = {Li, Sijia and Cui, Tianyu and Chen, Miao and Lin, Xinjie and Gu, Zheyuan and Deng, Xinhao and Xu, Ke and Li, Qi},
  title     = {From Length to Content: Token-Length Side-Channel Attacks on {LLM} {API} Merged Outputs},
  booktitle = {35th {USENIX} Security Symposium ({USENIX} Security 26)},
  year      = {2026},
  pages     = {2187--2206},
  publisher = {USENIX Association},
  url       = {https://www.usenix.org/conference/usenixsecurity26/presentation/li-sijia}
}
```

The official page's BibTeX at cached HTML lines 407–418 confirms every field, including the eight authors in their existing order. The published PDF title page confirms the title and author spellings; its first article page is 2187. Preserve the existing citation key. This removes the obsolete “Accepted for publication” note and corrects the word order “LLM API Merged Outputs.” No new literature or stronger empirical claim is introduced.
