# Integracja zakończonych obliczeń — 2026-11-01

Zmieniono wyłącznie `article1/template.tex`. Pliki wejściowe i operacje są zachowane w tym katalogu; PDF kompiluje agent nadrzędny. Autorzy, tytuł, afiliacje, deklaracje, czasopismo oraz klucze i kolejność cytowań pozostały bez zmian.

## Wyniki i interpretacja

- Główne metryki, bootstrapy promptów i scenariuszy, Wilson LR, EOS oraz wykres pochodzą z pełnych 2100 rekonstrukcji zweryfikowanych długości. Pięć niediagnostycznych serii ma średnie 0,4820–0,5355; archiwalne TinyLlama/Phi pozostają diagnostyczne (0,0792/0,1210). Istotnych jest 15/21 par w obu analizach. Wszystkie siedem przedziałów scenariuszowych jest szerszych.
- Osobna tabela nowych 300 referencji na model podaje TinyLlama 0,4352 i Phi-3.5-mini 0,5042, ich CI, ED, ROUGE, LR oraz EOS. Nie połączono ich z pozostałymi pięcioma seriami w nowy ranking.
- Uwzględniono kontrolowany replay przy znanych nowych token IDs: legacy decoding odtwarza dokładnie wszystkie 600 dawnych tekstów, fragmentów i wektorów; różnice renderowania obejmują wyłącznie spacje ASCII. Protokoły T5 i seeds są zgodne. Sparowane zmiany phi wynoszą 0,3560 [0,3396; 0,3731] i 0,3831 [0,3670; 0,3982], ze scenariuszowym bootstrapem. Jest to wspólny efekt zmiany wejścia i referencji; nie rozdzielono tych wkładów ani nie odzyskano historycznych IDs/wariantów wag.
- Zastąpiono osiem wierszy kontroli i ich opis: tylko 3/6 kontrastów spełnia próg Holma; oba cross-prompt są nierozstrzygające. Wyjaśniono różne seedy kontroli i obron dla tych samych źródłowych śladów.
- Zastąpiono 12 wierszy obron i wykres: redukcje 36,7–88,7%, 10/10 istotnych par. Random padding ma teraz rzeczywiście zapisany narzut 79,6%/85,5%, a wykres średni koszt 82,6%. Usunięto poprzednie wartości oczekiwane i ich gwiazdki.
- Wszystkie 22 komórki MAP mają aktualne bootstrapowe CI; odtworzone 3300 predykcji i pozostałe statystyki są dokładnie zgodne z poprzednim archiwum. Opisano rzeczywiste zapisane wejścia enkodera zamiast wyłącznie replay przygotowania wejścia.
- Usunięto wcześniejszą regresję i historyczne przykłady jakościowe z głównego wywodu, żeby nie mieszać ich z nową kampanią. Pozostałe archiwalne liczby dotyczą źródłowych odpowiedzi i audytu kolektora, a nie starszych rekonstrukcji.

## Metody i stan artykułu

Przepisano abstrakt, wprowadzenie, metody, wyniki, dyskusję, wnioski oraz opis artefaktów. Obserwacja jest jawnie instrumentowanym oracle'em logów; nie ma twierdzenia o pasywnej ekstrakcji TLS. Utrzymano rozróżnienie wszystkich segmentów / maksymalnie trzech / pierwszego segmentu, limity 80 T5, trzy próbki dla głównych i pilotowych rekonstrukcji oraz jedną dla MAP. Dodano przypięte nowe victim revisions, float16 i offload Phi oraz float32 T5. Wszystkie 62 757 + 8835 + 10 748 segmentów odpowiednich kohort zostały przetworzone, bez wcześniejszych zatrzymań.

Publiczna archiwizacja nowego pakietu jest oznaczona jako oczekująca; tekst nie sugeruje, że lokalne nowe pliki zostały już opublikowane na GitHub. Historycznych manifestów ani `completed_manifest.json` nie nadpisano.

## Kontrola

- Źródło gotowe do kompilacji: SHA-256 `76de97bf22e37d71dd348812f49fced10185bd39947ec12b7b27f8b38b531bdb`.
- Niezależna kontrola faktycznego TeX: **454 pola liczbowe, 85 wierszy w 10 tabelach oraz 2 wykresy; zero niezgodności**. Raport nadrzędny: `../manuscript_numeric_validation.json`.
- Brak brakujących/powtórzonych etykiet; cytowania i deklaracje zachowane; `git diff --check` przeszedł.
- PDF nie był kompilowany w tym podzadaniu. Agent nadrzędny wykonuje kompilację, kontrolę wizualną i końcowy snapshot; ewentualne późniejsze zmiany układu powinny otrzymać nowy hash i kontrolę liczbową.

`numeric_rows.json` zawiera wygenerowane wiersze tabel i dane wykresów. `integration_receipt.json` wiąże źródła liczb z plikiem TeX. `prose_operations.json`, `integrated.diff` i `matched_rendering_and_availability.diff` dokumentują edycje; `before/` zachowuje tekst wejściowy. Skrypty integracyjne są recepturami tej zmiany, a nie generatorami przeznaczonymi do ponownego uruchomienia po późniejszych poprawkach układu.
