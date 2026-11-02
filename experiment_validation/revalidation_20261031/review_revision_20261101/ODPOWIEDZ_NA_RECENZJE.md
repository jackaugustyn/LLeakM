# Zmiany po recenzji ARS z 1 listopada 2026

Wprowadzono zmiany z `reviews/ars_electronics_20261101/RECENZJA.md` na podstawie polecenia autora „Ok, wprowadź zmiany z recenzji”. Aktualny [artykuł PDF](../../../article1/template.pdf) ma 24 strony; [źródło LaTeX](../../../article1/template.tex) zachowuje wyniki ukończonej kampanii 6600 rekonstrukcji. Nowy tytuł brzmi **Lexical Reconstruction from Instrumented LLM Token-Length Traces: Reference Validation and Defense Evaluation**.

Najważniejsze rozstrzygnięcie dotyczy R1-W1. Zastosowano dopuszczony w recenzji wariant publikacji wyników **dokładnie wykonanej implementacji**, z jawnym ograniczeniem selektora, oraz dodatkowo przeprowadzono ograniczony audyt GPU. Nie przedstawiono starych tabel jako wyników naprawionego rankingu. Oryginalny kod, predykcje i zamrożone recenzje pozostawiono bez zmian. Nie wykonano pełnej kampanii z selektorem maskującym PAD po EOS.

## Odpowiedzi na uwagi

| Uwaga | Wprowadzona zmiana | Dowód / lokalizacja |
|---|---|---|
| EIC-W1 | Tytuł precyzuje pomiar leksykalnej rekonstrukcji z instrumentowanych śladów. Ujednolicono dokumenty zgłoszeniowe. | Strona tytułowa; `COVER_LETTER.md`, `SUBMISSION_SYSTEM.md`, `SUBMISSION_CHECKLIST.md`. |
| EIC-W2 | Dodano jedną tabelę pięciu protokołów: kohorty, referencje, liczebności, budżety ofiary, zakres segmentów, kandydaci, ziarna i odsyłacze do wyników. Rozróżniono liczbę rekonstrukcji od niezależnych odpowiedzi. | Tabela 1, `tab:protocol_overview`. |
| EIC-W3 | Przygotowano zamrożony lokalny pakiet S1 `LLeakM-S1-20261101-v1`, manifest SHA-256, README i weryfikator bez GPU. Wskazano dostęp przez materiały uzupełniające/autorów i nadal oczekującą publiczną archiwizację. | [Pakiet](../../../article1/supplement/LLeakM-S1-20261101-v1.tar.gz), [weryfikacja pakietu](../../../article1/supplement/LLeakM-S1-20261101-v1.verification.json), dodatek A i Data Availability. |
| R1-W1 | Opisano pełny sumator wyników przejść, PAD po EOS oraz fakt używania rozkładów po ograniczeniach próbkowania. Dodano audyt 24 wejść i 72 kandydatów. Ograniczenie jest jawne w metodach, rysunku 2, dyskusji i wnioskach. Wyjaśniono wyjątek dla jednokandydatowego MAP. | Dodatek C; [raport audytu](ranking_audit/audit_report.md), [pełni kandydaci](ranking_audit/results.jsonl). |
| R1-W2 | Zdefiniowano lokalną dominantę, 16 losowań kalibracyjnego paddingu, priory i DP dla niewidzianych krotek, reguły remisów, awaryjne powtórzenie dominanty oraz heurystykę rozmiaru ostatniej grupy. Wycofano sugestię globalnego MAP całego ciągu. | Dodatek B.1; [dowody metod](methods_evidence.json). |
| R1-W3 | Oddzielono strumień RNG głównego agregatora, ziarna komórek i kontrastów secondary oraz sparowany bootstrap renderowań. Podano funkcję BLAKE2s, argumenty, kodowanie i ziarna transformacji. | Metody statystyczne i dodatek B.2; niezależnie odtworzono wszystkie 3900 par ziaren secondary. |
| R2-W1 | Rozróżniono pełny ślad `Z` od zachowanego wejścia `E(Z)` w rekonstrukcji jednosegmentowej. Usunięto przypisywanie wyniku `pad_32` wykorzystaniu różnic długości, które nie docierają do dekodera. | Metody, wyniki obron/MAP i dyskusja; audyt wszystkich 660 rekordów, sześć stałych grup wejść. |
| R2-W2 | Wyniki 27% i około 53% u Weissa przypisano pierwszym segmentom i kosinusowi embeddingów MiniLM; podano progi i nieporównywalność z obecnym TF cosine/LR. | Related Work; [weryfikacja oryginałów](sources_proposal.json). |
| R2-W3 | Dodano udokumentowany trening T5-Large na pierwszych odpowiedziach UltraChat „Questions About the World”, role modeli first/middle oraz różnicę znaków i bajtów UTF-8. Udokumentowane karty modeli odróżniono od niezależnego audytu treningu. | Reconstruction Pipeline; wersjonowane karty modeli w `sources_proposal.json`. |
| DA-N1 | Podano 150/150 identycznych predykcji frozen `pad_32` przy wspólnych seedach między modelami. Wyjaśniono, dlaczego różne średnie wynikają tu z różnych referencji. | Wyniki MAP i [niezależna kontrola naukowa](validation/revision_science_check.json). |
| Bibliografia Li 2026 | Poprawiono tytuł, wpisano opublikowany status, strony 2187–2206 i adres oficjalnej publikacji. Zachowano klucz cytowania i autorów. | [Różnica bibliografii](bibliography.diff); oryginalny BibTeX USENIX. |

## Nowe obliczenia i ich granice

Audyt CUDA odtworzył **24/24** wcześniej zapisane pierwsze predykcje. Po sumowaniu tylko do pierwszego EOS, z włączeniem EOS i wykluczeniem późniejszego PAD, wybór tekstu zmienił się w **18/24** przypadkach. Z 72 kandydatów 38 miało ogon po EOS z co najmniej jednym wynikiem niefinitywnym; przed EOS i na EOS takich wyników nie było. Wszystkie kandydaty, token IDs, wyniki przejść, seedy i rankingi są zapisane. Sześć testów maskowania EOS przeszło pomyślnie.

Próba była celowo wybrana według skrajnych długości wejść oraz warunków obron. **18/24 nie jest estymatą częstości błędu w kampanii.** Nie przeliczano wpływu na pełne odpowiedzi, bo zmiana wybranego tekstu zmienia kontekst kolejnych segmentów. Wyniki, przedziały i wnioski w tabelach pozostają warunkowe względem pierwotnego selektora. Poprawiona pełna rekonstrukcja wymagałaby osobnej kampanii. 3300 prób MAP nie wykonuje wyboru między kandydatami.

Dokładne rachunki dla istniejących danych pozostały ważne. Powtórnie sprawdzono wszystkie 660 wejść `pad_32` i odtworzono ziarna wszystkich 3900 rekordów secondary. Specyfikacja MAP, bootstrapu oraz korekty literatury nie zmieniają danych ani nie wymagają nowej inferencji.

## Weryfikacja i odtwarzalność

- [Kontrola liczb](manuscript_numeric_validation.json): 13 tabel, 486 sprawdzonych pól tabel i danych wykresów, brak rozbieżności. Treść 12 istniejących tabel oraz dane dwóch wykresów numerycznych zachowane bez zmian.
- [Kontrola merytoryczna](validation/revision_science_check.md): brak nierozwiązanych sprzeczności w poprawkach; niezależne odtworzenie dowodów MAP, seedów, stałych wejść i selektora.
- [Kontrola PDF](final_build_validation.json) i [oględziny](validation/visual_review.md): 24 strony, 13 tabel, cztery rysunki; brak błędów kompilacji, nieokreślonych odwołań i przepełnień. Pozostały standardowe ostrzeżenia `hyperref` dotyczące matematyki w metadanych abstraktu.
- Pakiet S1 sprawdza się poleceniem `python3 verify_supplement.py` po rozpakowaniu. Weryfikator ponownie oblicza φ wszystkich 6600 rekonstrukcji i sprawdza 600 zapisów nowych ID, źródła, kompletność oraz hashe plików. Kontrola pakietu nie odtwarza inferencji modeli.
- [Poprawki w formacie JSON](revision_patch.json), [deterministyczny skrypt](apply_revision.py), [raport zastosowania](apply_report.json) i [diff](manuscript.diff) wiążą zmiany z zachowaną wersją wejściową. Abstraktu nie wygenerowano od nowa; wprowadzono w nim tylko dwa doprecyzowania zakresu implementacji i lokalnego MAP.

Jest to wdrożenie uwag z lokalnej symulowanej recenzji ARS-Codex i weryfikacja poprawionych materiałów, a nie nowa decyzja czasopisma ani formalny certyfikat zatwierdzenia ARS. Nie wykonano publikacji, wysyłki zgłoszenia ani publicznego przesłania artefaktów. Publiczna archiwizacja oraz decyzje zgłoszeniowe pozostają oddzielnymi działaniami autora.
