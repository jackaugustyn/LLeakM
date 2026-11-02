from pathlib import Path
import re
root = Path(__file__).resolve().parents[3]
p = root / 'article1/template.tex'
s = p.read_text()
figures = {
'fig:threat_model': r'''\begin{figure*}[ht]
\centering
\resizebox{0.96\textwidth}{!}{%
\begin{tikzpicture}[>=Latex,font=\small,
box/.style={draw,rounded corners,align=center,minimum height=12mm,text width=32mm,fill=blue!5},
obs/.style={box,fill=orange!12},eval/.style={box,fill=gray!10}]
\node[box] (prompt) at (0,0) {Synthetic prompt corpus};
\node[box] (victim) at (4.1,0) {Local victim model\\text generation};
\node[box] (records) at (8.2,0) {Logged text fragments\\and payload bytes};
\node[obs] (lengths) at (4.1,-1.8) {Validated trace $X$\\ordered byte lengths\\(UTF-8)};
\node[eval] (reference) at (8.2,-1.8) {Emitted reference $R$\\evaluator only};
\node[box,fill=green!6] (attacker) at (4.1,-3.6) {Fixed T5 reconstructor\\length input only};
\node[eval] (metrics) at (8.2,-3.6) {Lexical evaluation\\compare $\widehat R$ with $R$};
\draw[->] (prompt)--(victim);
\draw[->] (victim)--(records);
\draw[->] (records.south west)--(lengths.north east);
\draw[->] (records)--(reference);
\draw[->] (lengths)--(attacker);
\draw[->] (reference)--(metrics);
\draw[->] (attacker)--node[above] {$\widehat R$} (metrics);
\node[align=left,text width=31mm,font=\footnotesize] at (0,-2.6)
{The attacker receives neither the prompt nor reference text.\\[3pt]
Transport extraction and timing are outside this experiment.};
\end{tikzpicture}%
}
\caption{Information flow in the instrumented experiment. Logged fragments supply two separate evaluator products: a validated length trace and the emitted reference text. Only the lengths enter the reconstructor; the reference is used for scoring. The diagram makes no assumption that an instrumented write is a valid SSE event or an observable encrypted-transport record.}
\label{fig:threat_model}
\end{figure*}''',
'fig:pipeline': r'''\begin{figure*}[ht]
\centering
\resizebox{0.98\textwidth}{!}{%
\begin{tikzpicture}[>=Latex,font=\small,
box/.style={draw,rounded corners,align=center,minimum height=13mm,text width=33mm,fill=green!6}]
\node[box] (trace) at (0,0) {Validated logged\\payload lengths};
\node[box] (transform) at (4.4,0) {Length transformation\\then optional MAP inverse};
\node[box] (segment) at (8.8,0) {Heuristic segmentation\\of the resulting vector};
\node[box] (input) at (8.8,-2) {Segment + context\\serialize and tokenize\\encoder limit: 80 tokens};
\node[box] (decode) at (4.4,-2) {First- or middle-segment T5\\stochastic generation\\output limit: 80 tokens};
\node[box] (rank) at (0,-2) {Rank candidates\\select one text per segment};
\node[box,fill=gray!10] (output) at (0,-4) {Concatenate chosen text\\score against full reference};
\draw[->] (trace)--(transform);
\draw[->] (transform)--(segment);
\draw[->] (segment)--(input);
\draw[->] (input)--(decode);
\draw[->] (decode)--(rank);
\draw[->] (rank)--(output);
\draw[->] (rank.south east) -- (2.1,-3.1) --
node[below,align=center,font=\footnotesize] {Next segment: context is only\\the immediately preceding chosen text} (8.8,-3.1) -- (input.south);
\end{tikzpicture}%
}
\caption{Executed reconstruction pipeline. Unmodified conditions pass lengths directly to segmentation. Candidate selection precedes each context update; the first segment has no reconstructed context. Main and new-reference runs decode all segments with three samples per segment; 96-token pilots allow up to three segments with three samples each; the five-seed MAP study uses one sample for the first segment per seed. The 80-token encoder limit includes the instruction and context. All settings score their output against the full stored response.}
\label{fig:pipeline}
\end{figure*}''',
'fig:model_ci': r'''\begin{figure*}[ht]
\centering
\includegraphics[width=\textwidth]{figures/latest_model_results.pdf}
\caption{Completed all-segment reconstructions with scenario-bootstrap 95\% intervals ($N=300$ prompts in 60 scenarios per model; 512-token victim limit; three T5 samples per segment; 80-token T5 input and output limits). (a) Archived-reference results; $\dagger$ marks reference-format diagnostics. (b) Separate, matched rendering experiment: legacy decoding of newly recorded token IDs reproduces the two archived diagnostic inputs and references exactly. Full-sequence decoding changes both the trace and scoring reference under identical T5 settings and seeds. Annotated $\Delta\phi$ intervals resample paired scenarios. Marginal intervals match Tables~\ref{tab:scenario_sensitivity} and~\ref{tab:fresh_results}. The two panels do not form a pooled seven-model ranking.}
\label{fig:model_ci}
\end{figure*}''',
'fig:defense_tradeoff': r'''\begin{figure}[ht]
\centering
\includegraphics[width=\textwidth]{figures/latest_defense_tradeoff.pdf}
\caption{Mean lexical-score reduction versus realized protected-payload overhead, shown separately for the two 96-token pilot models ($n=30$ prompts each; up to three segments, three samples per segment, 80-token T5 input and output limits). Identical axes show every model--transformation result from Table~\ref{tab:defense_results}. Both coordinates use the saved completed runs, including the realized randomized padding. Batching has zero payload-sum overhead and changes the observable-unit count; transport overhead and latency were not measured. These are frozen-pipeline point estimates, not information-theoretic privacy estimates.}
\label{fig:defense_tradeoff}
\end{figure}'''
}
pattern = re.compile(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', re.S)
seen=[]
def replace(m):
 for key,value in figures.items():
  if r'\label{'+key+'}' in m.group():
   seen.append(key)
   return value
 raise RuntimeError('unexpected figure')
s=pattern.sub(replace,s)
assert sorted(seen)==sorted(figures)
p.write_text(s)
print('Replaced', seen)
