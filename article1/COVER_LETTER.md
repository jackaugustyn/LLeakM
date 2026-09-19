Dear Editors of *Electronics*,

Please consider our manuscript, “Reconstructing LLM Outputs from SSE-Equivalent Length Traces: A Cross-Model Study of Leakage and Defenses”, for publication as an Article in *Electronics*.

The manuscript presents a controlled study of information leakage from token-granular application-event lengths in streaming large language model services. Using 300 aligned prompts for each of seven open-weight victim models, we evaluate the transfer of a fixed T5-based reconstruction pipeline and quantify uncertainty with bootstrap confidence intervals and paired tests with family-wise error correction. We additionally evaluate five defense configurations based on bucketing, fixed-width framing, batching, and randomized padding, reporting both residual lexical recovery and operational overhead.

The work is relevant to the readership of *Electronics* because it addresses the security and privacy of deployed AI systems, application-layer streaming architectures, side-channel leakage, and practical countermeasures. We state the limits of the threat model explicitly: the experiment starts from an instrumented, token-granular application-length oracle and does not claim passive extraction of SSE event boundaries from TLS, HTTP/2, or QUIC traffic. This separation enables a reproducible assessment of the information carried by length traces while identifying transport-layer extraction and adaptive attacks as necessary future work.

The manuscript was prepared by Paweł Augustynowicz and Szymon Jędrzejczak, Faculty of Cybernetics, Military University of Technology, Warsaw, Poland. Paweł Augustynowicz (pawel.augustynowicz@wat.edu.pl) and Szymon Jędrzejczak (szymon.jedrzejczak@wat.edu.pl) are the corresponding authors.

The code, prompts, per-sample outputs, analysis scripts, and reproducibility manifest accompanying the manuscript are publicly available at https://github.com/jackaugustyn/LLeakM.

Before submission, the authors will confirm in the submission system that the manuscript is original, is not under consideration elsewhere, has been approved by both authors, and satisfies all applicable funding, conflict-of-interest, data-availability, and research-integrity requirements.

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
