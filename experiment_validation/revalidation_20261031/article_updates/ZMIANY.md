# Aktualizacja artykułu po przeliczeniu statystyk — 2026-10-31

Zakres: `article1/template.tex` i `experiment_validation/README.md`. Polecenie nadrzędne obejmuje poprawienie artykułu na podstawie zweryfikowanych danych oraz wznowienie obliczeń; te edycje dotyczą zakończonego przeliczenia statystyk. Nie wprowadzają wyników częściowego przebiegu CUDA.

## Zmiany

- Dodano osobny opis i tabelę eksploracyjnej analizy 60 scenariuszy: średnia pięciu wariantów, 10 000 bootstrapów scenariuszy, sparowane testy na średnich i korekcja Holma dla 21 par. Przepis i wszystkie siedem CI pochodzą z `statistics/scenario_cluster_sensitivity.json`. Te same 17 par spełniają próg istotności. Sześć CI jest szerszych, maksymalnie o 37,7%, a Phi-1.5 o 1,3% węższy.
- W tabeli obron zmieniono wyłącznie dwa historycznie nieodtwarzalne koszty random paddingu: 78,5% → **79,8% oczekiwane** i 81,9% → **83,1% oczekiwane**. Oznaczono je gwiazdką oraz wyjaśniono średnią `100*4T/B`, niespójne archiwalne koszty i nieodzyskiwalność dawnych realizacji. Punkt wykresu ma teraz x=81,4%, a y pozostaje archiwalną redukcją 63,4%; podpis jasno rozróżnia oba źródła.
- Historyczną implementację `iter_lines()` i niezależne dekodowanie tokenów opisano w czasie przeszłym. Dodano informację o obecnych naprawach: zweryfikowany log-length oracle, zgodność fragmentów z całym dekodowaniem i zapis token IDs. Nie poświadczono wstecznej poprawności archiwum.
- Snapshot z wcześniejszej rewizji nie jest już nazwany bieżącym stanem po naprawach. Dodano ścieżki odrębnych artefaktów rewalidacji i warunek kompletności nowych przebiegów.
- README otrzymał odpowiedni opis nowego kontraktu, tabelę scenariuszową, koszty oczekiwane, wynik przeliczenia metryk oraz poprawione wskazówki interpretacji nowych uruchomień.

## Kontrola

`validation.json` zawiera hashe końcowych plików i wyniki kontroli:

- zachowano wszystkie 18 poleceń cytowania w niezmienionej kolejności;
- w istniejących numerycznych wierszach tabel zmieniono tylko dwa koszty paddingu; dodano siedem wierszy osobnej tabeli scenariuszowej;
- siedem CI oraz oczekiwane koszty sprawdzono automatycznie względem JSON;
- brak powtórzonych etykiet i tytułów podsekcji; `git diff --check` przeszedł;
- PDF pozostał bez zmian, zgodnie z poleceniem nadrzędnym. Kompilacja i kontrola układu pozostają do wykonania po zakończeniu kolejnych zmian.

Kopie wejściowe: `before/`. Dokładne różnice: `template.diff`, `README.diff`. Nie zmieniano kodu wykonawczego ani danych historycznych.

## Pozostałe prace nadrzędne

Pełna kampania T5 i nowe generacje pozostają poza tym zestawem zmian. Gdy będą dostępne kompletne wyniki, konieczne będzie ponowne uzgodnienie tabel, ograniczeń i stanu artefaktów. Nadrzędny agent bada także faktycznie ukończone segmenty (żądanie wszystkich segmentów nie musi oznaczać ukończenia każdego z nich). Nie dodano nowego poświadczenia liczby segmentów. Audyt historycznych dawców `cross_prompt` przeprowadzony przez nadrzędnego agenta nie wykazał samodawców w wykorzystanym pilotażu, więc tej sekcji nie zmieniano.
