# Pełna kampania z selektorem do EOS

Autoryzacja autora: „Wykonaj pełną kampanię z poprawionym selektorem”. Kampania jest osobna od `revalidation_20261031`; poprzedni pakiet S1 i predykcje pozostają materiałem porównawczym. Polecenie obejmuje poprawienie kodu, testy, uruchomienie, monitorowanie i ponowne obliczenie statystyk. Nie obejmuje publikacji ani wysyłania materiałów na zewnątrz.

## Niezmienione wejścia i zakres

| Eksperyment | Rekonstrukcje | Referencje | Segmenty / kandydaci |
|---|---:|---|---|
| Główna kohorta archiwalna | 2100 | Te same 300 odpowiedzi każdego z siedmiu modeli | Wszystkie / 3 |
| Zweryfikowane TinyLlama | 300 | Istniejące nowe odpowiedzi z zapisanymi ID | Wszystkie / 3 |
| Zweryfikowane Phi-3.5 | 300 | Istniejące nowe odpowiedzi z zapisanymi ID | Wszystkie / 3 |
| Kontrole śladu | 240 | Istniejący pilot 96-tokenowy | Do trzech / 3 |
| Pilot obron | 360 | Istniejący pilot 96-tokenowy | Do trzech / 3 |
| MAP | 3300 | Istniejąca kohorta 512-tokenowa; 30 promptów, pięć seedów, 11 warunków, dwa modele | Pierwszy / 1 |

Łącznie: **6600 nowych rekonstrukcji T5**. Nie generuje się ponownie odpowiedzi modeli ofiar: zachowanie tych samych referencji i śladów umożliwia dopasowane porównanie reguły selekcji. Nie zmienia się podziałów kalibracji/testu, transformacji obronnych ani ich zapisanych realizacji. MAP odtwarza zapisane wejścia odwrotności; nie uczy jej ponownie.

Zachowane są zapisane seedy poszczególnych rekonstrukcji, maksymalnie 80 tokenów wejścia i wyjścia T5, `top_k=50`, temperatura 1, zakaz powtórzenia bigramu, float32 i CUDA. Checkpointy T5: first `057cf090b719f5cf89da504babd2122d404e0953`, middle `07e2b3fc96bc129eb4a2871e58731c07e889cecc`. Seedy ustawia się po załadowaniu modeli. Referencje ocenia się w całości, zgodnie z dotychczasowym protokołem.

## Jedyna planowana zmiana decyzji dekodera

Selektor sumuje znormalizowane wyniki wybranych tokenów ze zwróconych rozkładów próbkowania **do pierwszego wygenerowanego EOS włącznie**. Pomija wszystkie późniejsze pozycje PAD. Jeśli EOS nie wystąpi, zachowuje wszystkie kroki generacji. Prefiks dekodera nie jest tokenem wygenerowanym. Nie wprowadza się normalizacji długością. Deduplikacja tekstów i stabilne rozstrzyganie remisów pozostają zgodne z dotychczasową regułą.

To wyniki po ograniczeniach próbkowania, nie bezwarunkowe prawdopodobieństwa modelu. Błąd wyznaczania wyników lub wartość niefinitywna w zachowanym prefiksie zatrzymuje wykonanie, zamiast cicho zastępować ranking zerami. W każdej wielokandydatowej generacji zapisuje się wszystkie ID, teksty, wyniki przejść, położenie EOS, liczbę sumowanych kroków, nową sumę oraz historyczną sumę obejmującą ogon PAD. W ścieżce jednego kandydata nie włącza się `output_scores`; generowanie pozostaje identyczne.

Wybrany tekst jest kontekstem kolejnego segmentu, dlatego powtarza się całe odpowiedzi, nie tylko ich pierwsze segmenty. Zmiany dalszych kandydatów wynikające ze zmienionego kontekstu i przebiegu losowań należą do efektu całej poprawionej implementacji.

## Kontrole jakości i analizy

1. Testy selektora obejmują rzeczywisty kontrprzykład Transformers, EOS/PAD, brak EOS, wiele EOS ID, prefiks dekodera, remisy, duplikaty, błędy wyników i stan RNG.
2. Próba przed kampanią sprawdza pełne wielosegmentowe rekonstrukcje na identycznych zapisanych wejściach. Jej wyniki trafiają do osobnego katalogu `preflight/` i nie zwiększają liczebności kampanii.
3. Każdy rekord jest zapisywany trwale. Wznowienie sprawdza protokół, wejścia, hashe i kompletność audytu kandydatów. Wynik częściowy nie otrzymuje statusu ukończonego.
4. Po etapach inferencji niezależne walidatory sprawdzają źródła, unikalne klucze, tokenizację, segmenty, kontekst oraz metryki z zapisanych tekstów. Osobny audyt odtwarza maskowanie EOS i wybór kandydatów bez wywoływania selektora.
5. Wszystkie **3300 predykcji MAP powinny być identyczne** z poprzednią kampanią. Różnica wymaga wyjaśnienia, ponieważ ten protokół nie dokonuje wyboru między kandydatami.
6. Statystyki zachowują wcześniejsze jednostki resamplingu i rodziny korekt Holma. Nowe i poprzednie wyniki są porównywane w dopasowanych rekordach. Częstość zmian rankingu w tych samych nowych zestawach kandydatów jest oddzielna od porównania pełnych odpowiedzi między kampaniami.
7. Poprzedni audyt 24 przypadków jest dowodem diagnostycznym, a nie estymatą częstości w całej kampanii. Nowy audyt opisuje cały zrealizowany zbiór.

## Wykonanie i status

Kolejka uruchamia zadania sekwencyjnie, z trwałym statusem, kontrolą procesu, blokadą przed podwójnym uruchomieniem, hashami źródeł i limitem czasu na etap. `queue_plan.json` zawiera dokładne polecenia i warunki sukcesu. `queue/status.json` oraz statusy poszczególnych eksperymentów opisują rzeczywisty postęp. Błąd zatrzymuje kolejkę; późniejsze wyniki nie są oznaczane jako ukończone.

Po kompletnych obliczeniach `completed_results.json` i `completed_manifest.json` wiążą rezultaty z ich źródłami. Przed ukończeniem artykuł pozostaje dokumentem poprzedniej kampanii; nie podstawia się częściowych średnich do tabel.
