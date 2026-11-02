## Material Passport

- Origin Skill: ARS-Codex / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-01T16:09:12.328546+00:00
- Verification Status: VERIFIED — obliczenia i spójność artefaktów w zadeklarowanych protokołach
- Version Label: final_numeric_review_20261101

## Wynik kontroli

Niezależna kontrola zakończyła się powodzeniem. Wszystkie 754 pliki manifestu ukończonej kampanii miały zgodne hashe i rozmiary przed integracją artykułu. Sprawdzono wszystkie **6600 nowych rekonstrukcji**: 2100 głównych, 3900 kontroli/obron/MAP oraz 600 rekonstrukcji nowych, poprawnie dekodowanych referencji. Dodatkowe 600 generacji ofiar ma zapisane ID tokenów i osobne protokoły. Żaden z audytów końcowych nie uruchamiał inferencji.

Dla głównej kampanii i nowych referencji ponownie obliczono metryki całego tekstu oraz pierwszego zdania. Niezależne klasyczne programy dynamiczne w C++ dały dokładną zgodność ED i wszystkich miar ROUGE; największa różnica kosinusa wyniosła 2,22×10⁻¹⁶. W 3900 próbach dodatkowych wszystkie metryki były zgodne dokładnie. Odtworzono średnie, przedziały bootstrapu i testy sparowane.

## Najważniejsze liczby

| Seria | Model | Średnia φ | ED/ref | R1 precision | RL precision | LR > 0,5 | EOS |
|---|---|---:|---:|---:|---:|---:|---:|
| archiwalne referencje | qwen_1_5b | 0.535463 | 0.704223 | 0.292071 | 0.175999 | 196/300 | 167/300 |
| archiwalne referencje | qwen_3b | 0.525143 | 0.743135 | 0.255598 | 0.130182 | 189/300 | 50/300 |
| archiwalne referencje | llama_3_2_3b | 0.514621 | 0.746199 | 0.246364 | 0.131287 | 183/300 | 197/300 |
| archiwalne referencje | gemma_2_2b | 0.482001 | 0.774782 | 0.210103 | 0.103451 | 127/300 | 77/300 |
| archiwalne referencje | phi_1_5 | 0.516368 | 0.722868 | 0.256609 | 0.152357 | 170/300 | 144/300 |
| archiwalne referencje; diagnostyka | tinyllama_1_1b | 0.079159 | 0.880332 | 0.028196 | 0.021087 | 0/300 | 177/300 |
| archiwalne referencje; diagnostyka | phi_3_5_mini | 0.121017 | 0.878075 | 0.027185 | 0.020214 | 0/300 | 0/300 |
| nowe referencje | tinyllama_1_1b | 0.435205 | 0.757627 | 0.220877 | 0.126914 | 91/300 | 177/300 |
| nowe referencje | phi_3_5_mini | 0.504151 | 0.773299 | 0.223987 | 0.108394 | 157/300 | 0/300 |

W głównej analizie **15/21 par** pozostaje istotnych po Holm, zarówno przy jednostce promptu, jak i po uśrednieniu pięciu wariantów w każdym z 60 scenariuszy. Przedziały scenariuszowe są szersze dla wszystkich siedmiu modeli, maksymalnie o 41,6%. Żaden wynik główny, dodatkowy ani nowy nie jest dokładnie równy 0,5; zapis `>` i `>=` daje tu identyczne liczniki.

Kontrole mają **3/6** istotnych porównań po Holm; porównania cross-prompt są nieistotne dla obu modeli. Obrony mają **10/10** istotnych porównań i redukcje średniego wyniku 36,7031–88,7003%. Koszt losowego paddingu w nowych zapisanych realizacjach wynosi 79,6294% dla Qwen i 85,5198% dla Llama. To zarejestrowane koszty nowych prób, odmienne od wcześniej podanych wartości oczekiwanych.

**3300/3300 predykcji MAP** jest identycznych z archiwalnymi predykcjami; zatem 22 średnie i zmienność między ziarnami nie zmieniły się. Nowe granice bootstrapu wynikają z jawnego nowego przepisu losowania przedziałów. Zakres 22 średnich wynosi 0,042212–0,285244, a SD średnich między ziarnami 0,003227–0,018432.

Wszystkie 62 757 segmentów głównej kampanii zostały przetworzone; 1475 wywołań enkodera było obciętych do 80 tokenów. Nowe TinyLlama/Phi mają odpowiednio 8835/10 748 przetworzonych segmentów oraz 55/77 obciętych wywołań. Nie odnotowano wcześniejszego zatrzymania po pustym kandydacie.

## Ustalony defekt detokenizacji

Dla wszystkich 600 nowych zapisanych sekwencji ID odtworzono dawną detokenizację każdego tokenu oddzielnie. Uzyskano dokładnie te same historyczne teksty, fragmenty i wektory długości bajtowych w **300/300 prób TinyLlama i 300/300 Phi**. Dla tych samych ID pełne dekodowanie przywraca wyłącznie brakujące spacje: 82 460 w TinyLlama i 94 308 w Phi. Każdy zmieniony fragment różni się jedną przywróconą spacją na początku; po usunięciu spacji całe teksty są identyczne. Wszystkie nowe referencje zawierają spacje.

To bezpośrednio wyjaśnia różnice tekstów i długości dla ustalonych nowych sekwencji ID. Nie odzyskuje nieznanych historycznych ID ani wersji checkpointów. Sam wzrost metryk T5 nie stanowi ogólnego dowodu przyczynowego efektu detokenizacji lub porównawczej podatności modeli.

## Granice wnioskowania i kontrola interpretacji

Potwierdzone są numeryczne wyniki instrumentowanego eksperymentu dla zapisanych wejść. Nie zweryfikowano pasywnego odzyskiwania tych samych obserwabli z zaszyfrowanego ruchu produkcyjnego. Dwie historyczne serie nadal mają wadliwe referencje; nowe generacje należy prezentować osobno. T5 nadal obcina wejście do 80 tokenów, a transformacje zmieniają segmentację i pokrycie źródła. Kosinus i LR nie są skalibrowanymi miarami ujawnienia danych.

| Kontrola interpretacji | Wynik |
|---|---|
| Simpson | Modele mają te same prompty i równe wagi tematów; wyniki tematów pozostają opisowe. |
| Błąd ekologiczny | Średnie nie określają poprawności pojedynczej odpowiedzi ani częstości błędów semantycznych. |
| Berkson | Syntetyczny korpus ogranicza populację, na którą można przenosić wnioski. |
| Kolider/selekcja | EOS-only i regresje długości dotyczą wielkości po generacji; nie identyfikują efektów przyczynowych. |
| Częstości bazowe | LR jest częstością progu leksykalnego, bez kalibracji jako prawdopodobieństwo ujawnienia. |
| Regresja do średniej | Wspólne ziarna i powtórzenia promptów nie są nowymi niezależnymi jednostkami. |
| Przeżywalność | Zachowano wszystkie 300 rekordów każdej serii, także obcięte generacje. |
| Wielokrotne porównania | Sprawdzono rodziny testów i korekcje Holm; nie utożsamiono CI średniej z testem rang. |
| Wiele ścieżek analizy | Nowe analizy są odtworzeniem i eksploracją; brak poświadczenia prerejestracji. |
| Korelacja/przyczynowość | Wyniki pipeline’u pozostają warunkowe; tylko efekt dekodowania tekstu sprawdzono dla ustalonych ID. |
| Odwrócona przyczynowość | Kolejność transformacja–rekonstrukcja jest zapisana; związki cech i metryk pozostają opisowe. |

Pokrycie kontroli interpretacji: **11/11**. Szczegółowe liczby i przedziały do artykułu: `main_fresh_factledger.json` i `secondary_audit.json`. Dowody detokenizacji: `fresh_legacy_replay.json`. Walidacja manifestu: `completed_manifest_audit.json`.

## Dodatkowe porównanie sparowane i kontrola artykułu

Potwierdzono identyczność wersji bibliotek, kodu, wag T5, urządzenia, precyzji, liczby kandydatów i ziaren między główną a nową rekonstrukcją w każdej z 600 par. Różnią się konstrukcja wejściowego wektora i tekst referencyjny, wynikające z dwóch sposobów detokenizacji. To kontrolowane porównanie sposobu zapisu w tym pipeline, bez odzyskania historycznych ID ani uogólnienia na podatność produkcyjną.

- tinyllama_1_1b: średnia różnica φ = 0.356045; sparowany scenariuszowy 95% CI [0.339580; 0.373053].
- phi_3_5_mini: średnia różnica φ = 0.383134; sparowany scenariuszowy 95% CI [0.367047; 0.398213].

Przedziały różnic: 10 000 losowań bootstrapu po 60 sparowanych scenariuszach, seed 20261101, kolejność losowań zapisana w paired_detokenization_score_contrast.json.

Automatyczna kontrola rzeczywistego TeX potwierdziła **454 wartości liczbowe w 85 wierszach dziesięciu tabel i na dwóch wykresach**. Plik check_manuscript_numbers.py pozwala ponowić kontrolę po końcowej redakcji. Kontrola liczb nie zastępuje przeglądu prozy ani renderingu PDF.
