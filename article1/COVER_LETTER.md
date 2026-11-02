Dear Editors of *Electronics* and Guest Editors of the Special Issue “Trustworthy AI for Large Models and Security Systems”,

Please consider our manuscript, “Lexical Reconstruction from Instrumented LLM Token-Length Traces: Reference Validation and Defense Evaluation”, for publication as an **Article** in *Electronics* (ISSN 2079-9292), Section **Artificial Intelligence**, Special Issue **Trustworthy AI for Large Models and Security Systems**.

The manuscript evaluates a fixed Weiss-style T5 reconstructor using instrumented application-layer payload lengths. The completed EOS-bounded campaign comprises 6600 reconstructions: 2100 archived-reference responses from seven models, 600 responses from separate verified-reference cohorts, 240 trace-control trials, 360 defense trials, and 3300 first-segment reconstructions with five decoder seeds and an empirical MAP inverse. The 600 verified-reference responses and their token IDs were collected previously; this campaign reruns reconstruction on frozen inputs and does not generate new victim responses. Candidate selection sums sampling-processed transition scores through the first EOS, inclusive, and excludes later padding positions. All current inputs, candidate selections, outputs and metrics passed the campaign integrity checks.

The five non-diagnostic archived-reference series have mean lexical cosine scores of 0.5092–0.5768. The separate verified-reference means are 0.4677 for TinyLlama and 0.5269 for Phi-3.5-mini. Their independently decoded archived-reference series remain diagnostics and are not pooled with the verified-reference results into a seven-model vulnerability ranking. Within the EOS-bounded campaign, the matched reference-rendering contrast changes mean scores by 0.3845 and 0.3972, respectively; paired scenario-bootstrap intervals exclude zero. The contrast changes both the reconstruction input and scoring reference, so it does not identify their separate contributions.

The trace controls support a specific conclusion: shuffled and constant traces reduce the score significantly for both pilot models, whereas cross-prompt contrasts are not significant after Holm correction (four of six contrasts). All ten frozen-attacker defense contrasts are significant, with mean-score reductions of 43.8–89.6% and separately reported realized payload costs. The five-seed MAP study evaluates prompt-disjoint calibration of an inverse length mapping, rather than retraining T5. Both T5 input and output are capped at 80 tokens, and transformations affect segmentation and encoder coverage as well as the observed trace.

The study addresses a privacy concern in large-model systems: streaming metadata can expose token lengths even when payload contents are encrypted. The evidence is an instrumented length-oracle experiment; it does not establish extraction of application events from TLS, HTTP/2 or QUIC captures. Lexical overlap does not measure mutual information, semantic disclosure or production attack success. These deployment and measurement boundaries define the paper's contribution to trustworthy large models and security systems.

The study repository is https://github.com/jackaugustyn/LLeakM. Current campaign results and provenance are supplied in `experiment_validation/selector_eos_20261101/`, with the manuscript integration in `article_revision_20261102/`. Local Supplementary Data S2 (`LLeakM-S2-EOS-20261102-v1`) contains the current evidence, a SHA-256 manifest and a CPU-only verification entry point. Public archival of this version remains pending. Paweł Augustynowicz (pawel.augustynowicz@wat.edu.pl) and Szymon Jędrzejczak (szymon.jedrzejczak@wat.edu.pl), Faculty of Cybernetics, Military University of Technology, Warsaw, Poland, are the corresponding authors.

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
