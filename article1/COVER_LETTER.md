Dear Editors of *Electronics* and Guest Editors of the Special Issue “Trustworthy AI for Large Models and Security Systems”,

Please consider our manuscript, “Reconstructing LLM Outputs from SSE-Equivalent Length Traces: A Cross-Model Study of Leakage and Defenses”, for publication as an **Article** in *Electronics* (ISSN 2079-9292), Section **Artificial Intelligence**, Special Issue **Trustworthy AI for Large Models and Security Systems** (submission deadline 15 November 2026).

**Novelty.** The manuscript is a controlled, reproducible benchmark of leakage from token-granular application-event lengths in streaming LLM assistants, not a restatement of a single-model demonstration. Its contributions are: (i) an instrumented one-token-per-SSE-event collection protocol with a public artifact package; (ii) a cross-model evaluation of a fixed Weiss-style T5 reconstructor on 300 aligned prompts for each of seven open-weight models (2100 samples, 512-token budget), with bootstrap confidence intervals, paired Wilcoxon tests with Holm correction, and an EOS-complete versus truncated sensitivity table; (iii) a defense pilot that reports both residual lexical recovery and byte/frame overhead for bucketing, fixed-width framing, batching, and randomized padding; and (iv) a five-seed, prompt-disjoint empirical MAP inverse (22 complete cells, 3300 reconstructions) on 512-token traces of two models.

**Fit with *Electronics* and the Special Issue.** The Special Issue targets trustworthy large models and security systems. The paper studies a deployment-facing privacy failure mode of LLM assistants: application-layer streaming metadata can encode token lengths even when payload content is encrypted. It therefore belongs with work on the security of large-model systems, side-channel leakage, and practical countermeasures, which are core concerns of *Electronics* and of this Special Issue.

**Threat model.** The experiment starts from an optimistic length oracle: an instrumented service emits one SSE application event per generated token, and the attacker receives only the ordered event lengths. These laboratory traces are SSE-equivalent in framing and granularity. They are not TLS, HTTP/2, or QUIC records recovered from a network capture. We state this limit throughout the manuscript so that the results are read as a measure of the information in token-granular lengths, not as proof that those lengths can be extracted from encrypted transport traffic. Transport-layer extraction and adaptive attacks remain necessary future evaluations.

The complete artifact package is available at https://github.com/jackaugustyn/LLeakM. Paweł Augustynowicz (pawel.augustynowicz@wat.edu.pl) and Szymon Jędrzejczak (szymon.jedrzejczak@wat.edu.pl), Faculty of Cybernetics, Military University of Technology, Warsaw, Poland, are the corresponding authors.

Thank you for considering our manuscript.

Sincerely,

Paweł Augustynowicz  
Faculty of Cybernetics  
Military University of Technology  
gen. Sylwestra Kaliskiego 2  
00-908 Warsaw, Poland  
pawel.augustynowicz@wat.edu.pl  

Szymon Jędrzejczak  
Faculty of Cybernetics  
Military University of Technology  
gen. Sylwestra Kaliskiego 2  
00-908 Warsaw, Poland  
szymon.jedrzejczak@wat.edu.pl
