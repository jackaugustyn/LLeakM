# Obliczenia zakończone — 1 listopada 2026

Wszystkie 12 etapów kolejki zakończyło się **1 listopada 2026 o 16:15 czasu polskiego**. Nadzorca i procesy modeli zakończyły pracę; nie należy ich ponownie uruchamiać, aby sprawdzić status.

## Kompletność

- 2100/2100 rekonstrukcji głównych z poprawnych wektorów długości zapisanych w logach.
- 240/240 kontroli śladu, 360/360 obron i 3300/3300 prób MAP.
- TinyLlama: 300/300 nowych odpowiedzi, 300/300 rekonstrukcji i pełna weryfikacja.
- Phi-3.5: 300/300 nowych odpowiedzi, 300/300 rekonstrukcji i pełna weryfikacja.
- Łącznie **6600 rekonstrukcji T5 oraz 600 nowych generacji ofiar**.

[Stan kolejki](queue/status.json), [raport wyników](WYNIKI_WZNOWIENIA.md), [pełny wynik maszynowy](completed_results.json). Wszystkie trzy weryfikatory głównych/nowych rekonstrukcji mają `VERIFIED_COMPLETE`, a dodatkowe 3900 prób przeszło walidację całego projektu i ponowne obliczenie metryk.

## Najważniejsze wyniki

Średnie φ w nowej rekonstrukcji zachowanych referencji: Qwen1.5 0,535463; Qwen3 0,525143; Llama 0,514621; Gemma 0,482001; Phi1.5 0,516368; TinyLlama 0,079159; Phi3.5 0,121017. Dwie ostatnie serie nadal mają wadliwe historyczne referencje i są diagnostyczne.

Nowe, oddzielnie wygenerowane i poprawnie dekodowane referencje dają TinyLlama **0,435205** i Phi3.5 **0,504151**. Nie łączy się tych kohort z historycznymi referencjami pięciu pozostałych modeli w jeden ranking. Rewizje wag i środowisko są jawne, ale nie ustalono tożsamości historycznych token IDs.

Wynik testów modeli zmienił się z historycznych 17/21 na **15/21** istotnych par (zarówno przy jednostce promptu, jak i scenariusza). W nowych kontrolach istotne po Holm są **3/6** kontrastów; nie potwierdzono ogólnej przewagi prawdziwego śladu nad każdym wariantem kontrolnym. W obronach istotne są **10/10** kontrastów. Pełne liczby, efekty i przedziały znajdują się w JSON, a ich interpretacja wymaga zachowania opisanych zakresów wejścia i referencji.

## Pochodzenie i dalsza integracja

`completed_manifest.json` jest snapshotem z chwili zakończenia kolejki, **przed integracją nowych wyników do artykułu**. Niezależny przegląd sprawdził jego 754 pliki; szczegóły w `final_review_20261101/completed_manifest_audit.json`. Późniejsze zmiany dokumentów mają osobny stan i manifest; nie zastępują wcześniejszego poświadczenia.

Końcowy audyt naukowy i integracja artykułu są zakończone. Raport: `final_review_20261101/WERYFIKACJA_KONCOWA.md`; PDF: `article1/template.pdf` w katalogu głównym projektu (19 stron). Sprawdzono 454 pola liczbowe, 10 tabel i 2 wykresy; końcowa kompilacja nie zgłasza błędów ani nadmiarowych linii. Osobny `final_review_20261101/final_snapshot.json` wiąże bieżące dokumenty i dane po integracji. Nowe obliczenia zachowują ograniczenie 80 tokenów enkodera i dekodera, różne zakresy segmentów między eksperymentami i instrumentowany oracle długości. Wyniki leksykalne nie mierzą bezpośrednio ujawnienia danych ani skuteczności przechwytywania ruchu TLS.

Dawne `WZNOWIENIE.md`, `pause_state.json` i `resume_state.json` dokumentują wcześniejszą pauzę/wznowienie. Nie są aktualnym statusem wykonania.
