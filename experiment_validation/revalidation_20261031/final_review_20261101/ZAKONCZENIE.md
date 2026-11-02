# Zakończenie przeliczeń i aktualizacji artykułu

Wszystkie obliczenia zakończono **1 listopada 2026 o 16:15 czasu polskiego**: 6600 rekonstrukcji T5 i 600 nowych generacji ofiar. Następnie wykonano niezależny audyt wszystkich wyników i uzgodniono z nimi artykuł, dokumenty zgłoszeniowe i README.

Najważniejsze zmiany: nowe rekonstrukcje główne dają 15/21 istotnych par modeli; kontrolne kontrasty potwierdzono w 3/6 przypadków, a obrony w 10/10. Nowe referencje TinyLlama/Phi-3.5 dają średnie φ 0,435205/0,504151. Są prezentowane osobno. Replay 600 zapisanych sekwencji IDs dokładnie odtwarza dawnymi zasadami wszystkie archiwalne fragmenty, teksty i wektory długości; pełne dekodowanie przywraca wyłącznie spacje. Dopasowane porównanie renderowania zmienia jednocześnie wejście T5 i referencję oceny.

Wszystkie 3300 predykcji MAP odtworzono identycznie. Pozostałe metryki, efekty i przedziały sprawdzono niezależnymi obliczeniami. Artykuł ma aktualne tabele, wykresy, abstrakt, metody i wnioski. Usunięto stare nieaktualne twierdzenia i nie użyto uszkodzonych referencji jako dowodu niskiej podatności normalnie dekodowanych modeli.

Końcowy PDF ma **19 stron**. Kontrola **454 pól liczbowych w 10 tabelach i 2 wykresach** przeszła; kompilacja i inspekcja układu również. Raporty: `WERYFIKACJA_KONCOWA.md`, `manuscript_numeric_validation.json`, `manuscript/final_build_validation.json`, `integration_status.json`.

Nowe źródło i PDF pozostają lokalne; publikacji lub zgłoszenia nie wykonano. Metryki opisują instrumentowany eksperyment, nie potwierdzają przechwytywania TLS ani nie mierzą bezpośrednio ujawnienia danych. Brak oryginalnych historycznych token IDs pozostaje jawny, mimo dokładnej zgodności replayu zapisanych obserwacji.
