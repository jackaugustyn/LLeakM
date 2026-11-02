Dear Editors of *Electronics* and Guest Editors of the Special Issue “Trustworthy AI for Large Models and Security Systems”,

Please consider our manuscript, “Information Leakage from Token-Length Traces in Streaming LLMs: Cross-Model Reconstruction and Defense Evaluation”, for publication as an **Article** in *Electronics* (ISSN 2079-9292), Section **Artificial Intelligence**, Special Issue **Trustworthy AI for Large Models and Security Systems**.

**Contribution and evidence scope.** The manuscript evaluates a fixed Weiss-style T5 reconstructor in an instrumented length-oracle setting. A completed rerun reconstructs 2100 archived responses from verified logged payload lengths, repeats 240 trace-control and 360 defense trials, and evaluates 3300 first-segment reconstructions with five decoder seeds and an empirical MAP inverse. An additional 600 newly generated TinyLlama and Phi-3.5-mini responses have recorded token IDs and verified full-sequence decoding, followed by 600 T5 reconstructions. All recorded inputs, outputs and metrics passed completeness and integrity checks.

The results distinguish three sources of evidence: reconstruction on retained references, two-model pilot and MAP experiments, and fresh responses with verified decoding. A fixed-ID replay reproduces all 600 archived emitted texts and length vectors with independent-token decoding, while full-sequence decoding restores word boundaries. New TinyLlama and Phi-3.5-mini mean lexical cosine scores are 0.4352 and 0.5042; their malformed archived reference series remain diagnostic and are not pooled with the fresh results into a seven-model ranking. The rerun supports a narrower trace-control conclusion than the original analysis: three of six control contrasts are significant after Holm correction. Both T5 input and output are capped at 80 tokens, so segmentation and retained encoder coverage remain part of the measured pipeline. The matched rendering comparison retains the T5 code, weights, sampling settings and per-record seeds while changing both the length input and scoring reference. Mean lexical scores increase by 0.3560 for TinyLlama and 0.3831 for Phi-3.5-mini, with paired scenario-bootstrap intervals excluding zero. This joint effect does not separate the input and reference contributions or establish an intrinsic model-vulnerability ranking.

**Fit with *Electronics* and the Special Issue.** The Special Issue targets trustworthy large models and security systems. The paper studies a deployment-facing privacy failure mode of LLM assistants: application-layer streaming metadata can encode token lengths even when payload content is encrypted. It therefore belongs with work on the security of large-model systems, side-channel leakage, and practical countermeasures, which are core concerns of *Electronics* and of this Special Issue.

**Threat model.** The observation is an application-layer payload-length oracle. New inference consumes exact, validated length vectors; the paper also documents the archived line-parser defect and unavailable historical inputs. The experiment does not recover TLS, HTTP/2 or QUIC event boundaries from network captures. Lexical overlap does not measure mutual information, semantic disclosure or production attack success. Transport-layer extraction and end-to-end attacker retraining remain separate evaluations.

The study repository is https://github.com/jackaugustyn/LLeakM. Completed revalidation outputs and their provenance are supplied in `experiment_validation/revalidation_20261031/`; historical artifacts are preserved separately. New manuscript and supplement changes still require inclusion in the public submission package. Paweł Augustynowicz (pawel.augustynowicz@wat.edu.pl) and Szymon Jędrzejczak (szymon.jedrzejczak@wat.edu.pl), Faculty of Cybernetics, Military University of Technology, Warsaw, Poland, are the corresponding authors.

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
