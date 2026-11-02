# Weryfikacja naukowa zmian z recenzji

**Wynik: poprawki merytoryczne zgodne z dowodami.** Sprawdzenie dotyczy implementacji recenzji; nie jest nową decyzją redakcyjną ani potwierdzeniem wyników poprawionego selektora. Pakowanie i kontrola finalnego PDF pozostają osobnymi czynnościami.

Sprawdzony SHA-256 TeX: `9e7585d5867034eb7725b538c9499531c64b8e925afbf33e91e430d56556cc4b`.

| Uwaga | Wynik | Ocena |
|---|---|---|
| EIC-W1 | PASS | Title names lexical reconstruction, instrumented traces, reference validation and defense evaluation. |
| EIC-W2 | PASS | Five-row protocol map has correct cohorts, counts, budgets, segment/sample scopes and seed references; model-caption wording corrected. |
| EIC-W3 | CONTENT_PASS_PACKAGE_PENDING | Versioned local S1 and access route are described without claiming public archival; final archive and hashes are to be checked by the separate packaging verifier. |
| R1-W1 | PASS | Exact original PAD-inclusive selector is explicit in methods, diagram caption, limitations and conclusions; audit is purposive and first-segment-only; no corrected full-response results claimed; MAP exception stated. |
| R1-W2 | PASS | Local modes, support, add-one prior, tuple DP, tie rules, 16 padding draws, unattainable-sum fallback and final-group count heuristic match implementation. |
| R1-W3 | PASS | Bootstrap streams are separated correctly; decoder/transformation H formula independently matches all 3900 secondary records. |
| R2-W1 | PASS | Single-segment E(Z) distinction and constant-input inference supported by independent recomputation of all660 pad records; wording unique inputs corrected. |
| DA-N1 | PASS | 150 common-seed frozen pairs have exactly equal predictions; different mean scores correctly ascribed to different references. |
| R2-W2 | PASS | Weiss first-segment scope and MiniLM thresholds >0.9/>0.5 accurately distinguished from current TF cosine/full-response LR. |
| R2-W3 | PASS | Pinned model-card provenance reported as documented, not independently audited; UltraChat/first-middle roles and character versusUTF8 distinction accurate. |
| Li bibliography | PASS | Only li2026length entry changed; published title, author list, venue, pages2187--2206 andURL match cached official source. |

Niezależnie sprawdzono, że treść danych wszystkich 12 dotychczasowych tabel jest identyczna, podobnie jak dwa numeryczne rysunki. Nowa tabela zawiera wyłącznie zestawienie protokołów. Kod oryginalnego selektora nie zmienił się.

Ponowne zliczenia z rekordów potwierdzają 660 przypadków `pad_32`, stały zachowany encoder input w każdej z sześciu grup oraz identyczne predykcje wszystkich 150 sparowanych przypadków frozen MAP. Niezależne zliczenie kandydatów potwierdza 24 odtworzone pierwsze predykcje, 18 zmian selekcji w 72 kandydatach, 38 ogonów po EOS z wartościami niefinitywnymi i brak takich wartości przed/na EOS. Formuła H odtwarza ziarna decoderów i transformacji wszystkich 3900 rekordów secondary.

W trakcie sprawdzenia zgłoszono i potwierdzono poprawienie dwóch drobnych sprzeczności: podpis protokołów nie obejmuje już Tiny/Phi stwierdzeniem o modelach Qwen/Llama, a „unique inputs” zastąpiono indeksowanymi promptami, odróżniając je od identycznych wejść enkodera. Nie pozostały nierozwiązane uwagi merytoryczne do tych edycji.

Zamknięcie R1-W1 odbywa się w wariancie dopuszczonym przez recenzję: jawnie opisane wyniki pierwotnej implementacji oraz ograniczony audyt. Nie oznacza to naprawienia pełnej kampanii ani ustalenia częstości/długofalowego wpływu błędu.
