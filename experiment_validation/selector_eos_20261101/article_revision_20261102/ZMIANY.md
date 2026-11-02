# Artykuł oparty na zakończonej kampanii EOS

Wszystkie wyniki rekonstrukcji w `article1/template.tex` i `article1/template.pdf` pochodzą teraz z zakończonej kampanii `selector_eos_20261101`: **6600 rekonstrukcji z poprawionym selektorem**. W tej kampanii wykorzystano zamrożone odpowiedzi modeli ofiar, referencje i wejściowe długości. Nie generowano ponownie odpowiedzi modeli ofiar.

Zaktualizowano abstrakt, metodę wyboru kandydatów, wyniki, dyskusję, ograniczenia, wnioski, dodatki i dostępność danych. Wszystkie 13 tabel mają źródła w bieżącej kampanii. Oba wykresy liczbowe wygenerowano ponownie, diagram rekonstrukcji pokazuje sumowanie ocen do pierwszego EOS, a diagram przepływu informacji zachowuje poprawny model obserwacji. Dokumenty zgłoszeniowe wskazują bieżące wyniki i nowy suplement S2.

## Najważniejsze wyniki

| Zakres | Wynik bieżącej kampanii |
|---|---|
| Pięć głównych serii bez rozpoznanej wady formatu referencji | Średnia φ 0,5092–0,5768; LR@0,5 55,00–76,67% |
| TinyLlama, osobno zweryfikowane referencje | Średnia φ 0,4677; LR@0,5 44,33% |
| Phi-3.5-mini, osobno zweryfikowane referencje | Średnia φ 0,5269; LR@0,5 62,33% |
| Porównania modeli | 18/21 istotnych po korekcji Holma, zarówno dla promptów, jak i scenariuszy |
| Kontrole śladu | 4/6 istotnych; przetasowanie i stały ślad obniżają wynik obu modeli |
| Pilotaż obron | 10/10 istotnych; redukcja średniej φ 43,8–89,6% |
| Badanie MAP | 3300 bieżących rekonstrukcji; pierwszy segment, jeden kandydat, pięć ziaren |

Oba kontrasty `cross_prompt` pozostają nierozstrzygające po korekcji Holma. Ich nominalne przedziały bootstrapowe średniej i skorygowane testy rangowe odnoszą się do różnych wielkości i innego sposobu uwzględnienia wielokrotnych porównań; tekst opisuje tę różnicę.

W głównej części wykonano wszystkie 62 757 prób segmentów; 62 755 dało niepusty tekst. Dwa puste zakończenia dotyczą Llama i Phi-1.5. Nie pominięto żadnego dalszego segmentu. Osobne zweryfikowane kohorty dały 8835 i 10 748 niepustych wyników segmentów.

## Podstawa wnioskowania

Artykuł nie wykorzystuje wcześniejszych przewidywań rekonstruktora ani porównań ze starym selektorem. Archiwalne logi i odpowiedzi pozostają źródłami obserwacji. Sparowane porównanie dwóch sposobów dekodowania tokenów wykorzystuje w obu ramionach nowe rekonstrukcje EOS. Zmienia jednocześnie długości wejściowe i tekst referencyjny, więc nie rozdziela tych dwóch efektów.

Zachowano ograniczenia naukowe: wyniki opisują podobieństwo leksykalne, nie semantyczną poprawność ani skuteczny atak na ruch TLS. Dwie serie o wadliwych granicach słów są diagnostyczne. Transformacje wpływają również na segmentację i pokrycie wejścia T5, a kalibracja MAP jest rozłączna po dokładnym prompcie, lecz nie po scenariuszu.

## Weryfikacja i odtwarzanie

- `manuscript_numeric_validation.json`: 486 sprawdzonych wartości — 433 w tabelach i 53 w danych wykresów — oraz 121 sprawdzeń opisowych i powiązań artefaktów.
- `scientific_review.json`: niezależna kontrola abstraktu, metod, wyników, wniosków i interpretacji względem bieżących danych.
- `fixed_id_rendering_replay.json`: ponowne odtworzenie obu sposobów dekodowania 600 zapisanych sekwencji tokenów i powiązanie z bieżącymi rekordami.
- `campaign_integrity_check.json`: zgodność wszystkich 84 plików zamrożonego manifestu kampanii.
- `visual_review.json`: kontrola składu PDF, tabel, diagramów i wykresów, powiązana z SHA-256 ocenionego dokumentu.
- `final_validation.json`: spójność końcowego TeX, PDF, raportów i zachowanych deklaracji autorów.

Suplement `article1/supplement/LLeakM-S2-EOS-20261102-v1.tar.gz` zawiera bieżący artykuł, wyniki i kandydatów, niezbędne źródła, skrypty analiz, generatory tabel i wykresów oraz weryfikator. Końcowy raport weryfikacji rozpakowanej kopii znajduje się obok archiwum w pliku `.verification.json`. Sprawdzenie obejmuje wszystkie 6600 wyników φ, 86 788 wywołań dekodera i 253 764 kandydatów; opcjonalny tryb z dołączonymi tokenizerami działa bez wag modeli.

Archiwalne wyniki i wcześniejszy suplement S1 zachowano jako osobne artefakty. Nowy pakiet pozostaje lokalny; ta aktualizacja nie stanowi publikacji, wysłania artykułu ani potwierdzenia aktualnych wymogów czasopisma.
