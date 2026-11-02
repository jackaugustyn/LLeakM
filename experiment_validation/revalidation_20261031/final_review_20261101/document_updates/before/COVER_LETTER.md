Dear Editors of *Electronics* and Guest Editors of the Special Issue “Trustworthy AI for Large Models and Security Systems”,

Please consider our manuscript, “Information Leakage from Token-Length Traces in Streaming LLMs: Cross-Model Reconstruction and Defense Evaluation”, for publication as an **Article** in *Electronics* (ISSN 2079-9292), Section **Artificial Intelligence**, Special Issue **Trustworthy AI for Large Models and Security Systems**.

**Contribution and evidence scope.** The manuscript examines a fixed Weiss-style T5 reconstructor using archived outputs from seven open-weight models (300 aligned prompts per model; 2100 samples; a 512-token generation budget), together with 96-token trace-control and defense pilots. It reports lexical scores, bootstrap intervals, paired Wilcoxon tests with Holm correction, and results stratified by whether the victim response reached EOS. A separate five-seed study evaluates an empirical MAP inverse on the first heuristic trace segment, scored against the full reference response (22 settings; 3300 reconstructions). The revision distinguishes these protocols and documents how segmentation and the 80-token T5 input limit affect the observations available to the decoder.

The archived primary scores remain conditional on unresolved collection provenance: replaying the current line-based collector changes logged token-length sequences, and the exact historical reconstructor inputs were not retained. TinyLlama and Phi-3.5-mini are retained as apparatus diagnostics because their reference text has unresolved detokenization defects. The paper therefore does not present the seven-model score ordering as a validated comparison of transfer on ordinary model responses. Corrected reconstruction and reference-validation experiments remain outstanding.

**Fit with *Electronics* and the Special Issue.** The Special Issue targets trustworthy large models and security systems. The paper studies a deployment-facing privacy failure mode of LLM assistants: application-layer streaming metadata can encode token lengths even when payload content is encrypted. It therefore belongs with work on the security of large-model systems, side-channel leakage, and practical countermeasures, which are core concerns of *Electronics* and of this Special Issue.

**Threat model.** The intended observation is an optimistic application-layer length oracle: an instrumented service emits one event per generated token. The manuscript distinguishes intended token-byte lengths, direct log-derived traces used in the pilots and MAP study, and the current line-based collection path. These are laboratory observations, not TLS, HTTP/2, or QUIC records recovered from a network capture. The scores describe lexical overlap under the evaluated pipeline; they do not measure mutual information, semantic disclosure, or production attack success. Transport-layer extraction and end-to-end attacker retraining remain separate evaluations.

The study repository is https://github.com/jackaugustyn/LLeakM. The revision identifies gaps in historical input, code-version, and seed provenance rather than treating a current checksum snapshot as evidence of the historical execution environment. Paweł Augustynowicz (pawel.augustynowicz@wat.edu.pl) and Szymon Jędrzejczak (szymon.jedrzejczak@wat.edu.pl), Faculty of Cybernetics, Military University of Technology, Warsaw, Poland, are the corresponding authors.

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
