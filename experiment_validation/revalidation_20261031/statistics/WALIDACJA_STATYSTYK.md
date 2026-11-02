## Material Passport

- Origin Skill: ARS-Codex / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-31
- Verification Status: VERIFIED — ponowne obliczenie statystyk **warunkowo względem zachowanych tekstów**, bez potwierdzenia poprawności rejestracji i detokenizacji.
- Version Label: archived_statistics_revalidation_v1

## Wynik

Ponownie obliczono metryki wszystkich **2100 głównych par tekstów**, dodatkowo ich **2100 par pierwszych zdań**, **240 rekordów kontroli**, **420 rekordów obron** (360 obecnego protokołu i 60 historycznego `pad_fixed`) oraz **3300 rekordów MAP**. Nie wykonano inferencji modeli. Zachowane wartości ED i ROUGE oraz φ w kontrolach, obronach i MAP są dokładnie zgodne z obliczeniami: maksymalna różnica wynosi **0**. φ głównej kampanii przeliczono tym samym jednoznacznym słownym kosinusem; historyczne pola φ z różnych backendów nie były traktowane jako porównywalne.

**Werdykt całościowy: PARTIALLY_REPRODUCIBLE.** Statystyki leksykalne są reprodukowalne. Dwa raportowane narzuty losowego paddingu nie odtwarzają się z udokumentowanego przepisu ziaren. Ponadto poprawna arytmetyka nie usuwa problemów kolektora i zdeformowanych referencji opisanych w recenzji.

| Zakres | Wynik ponownego obliczenia |
|---|---|
| Siedem modeli, metryki główne, przedziały, LR i EOS | Wszystkie wartości `model_stats.json` zgodne dokładnie. |
| 21 testów par | Odtworzenie dawnego kodu zgodne dokładnie; niezależna metoda SciPy/rangi daje te same decyzje **17/21**, z prawidłowymi niezerowymi małymi wartościami p. |
| Regresja kowariatów | Współczynniki i R² zgodne dokładnie: pełny model ≈0,904, wskaźniki modeli ≈0,897, same kowariaty ≈0,749. |
| Macierz tematów | Wszystkie średnie zgodne dokładnie. |
| Kontrole śladu | Wszystkie średnie, przedziały i różnice zgodne dokładnie po naprawie błędu kodu agregacji. |
| Obrony | Wszystkie wyniki leksykalne, przedziały i liczby ramek zgodne; **10/10** porównań z baseline istotnych po Holm. Narzut `rand_pad_8` opisano poniżej. |
| MAP | Wszystkie wartości liczbowe 22 ustawień i kontrastów zgodne dokładnie. Sprawdzono pełny iloczyn 30 promptów × 5 ziaren, brak duplikatów i wspólne ziarna między warunkami. Kopia wyniku zawiera jedynie dodatkowe pole metadanych `config.protocol_id`. |

Odległość Levenshteina i LCS obliczono niezależnymi algorytmami bitowymi, porównanymi z oryginalnym programowaniem dynamicznym w **605** przypadkach, obejmujących puste ciągi, Unicode i losowe dane. Nie ograniczono kontroli do ponownego uśrednienia wcześniej zapisanych metryk.

## Rzeczywiste niezgodności i poprawki kodu

1. `trace_baselines.py` próbował wykonać `float()` na dwuelementowej tablicy percentyli. Takiego kodu agregacji nie dało się uruchomić. Naprawiono rozpakowanie obu wartości i sprawdzono rzeczywistą ścieżkę wznowienia z całkowicie zabronioną inferencją.
2. `analyze_stats.py` i `defense_eval.py` wyznaczały ogon rozkładu jako `1 − CDF`, co zaokrąglało dziesięć głównych wartości p do zera. Zastosowano `erfc`. Dla kontrastów z pełną separacją otrzymuje się np. p ≈ **6,1141×10⁻⁵¹**, a nie matematyczne zero. Wszystkie wcześniejsze decyzje przy 0,05 pozostają takie same. Naprawiono też zbędną korektę ciągłości przy dokładnie równych sumach rang dodatnich i ujemnych; w takim przypadku p wynosi 1.
3. Ziarna losowego paddingu zależą od `hash()`. Dwa różne historyczne pliki podają różne narzuty dla tych samych zapisanych rekonstrukcji. Przepis z `PYTHONHASHSEED=0` nie potwierdza historycznych realizacji paddingu. Nie zastąpiono starych wartości nową realizacją losową i nie uznano jej za odzyskaną prawdę historyczną.

| `rand_pad_8`: średni narzut jednostek payload | Qwen2.5-1.5B | Llama-3.2-3B |
|---|---:|---:|
| Raportowane w `defense_eval.json` / artykule | 78,528659% | 81,917065% |
| Zapisane w `defense_overhead.json` | 79,312100% | 82,665922% |
| Obecny replay przepisu, `PYTHONHASHSEED=0` | 79,716287% | 83,052767% |
| **Analityczna wartość oczekiwana** | **79,755339%** | **83,133614%** |

Ostatni wiersz jest wiarygodny jako **wartość oczekiwana**, nie zmierzony narzut historycznego uruchomienia. Dla śladu z T tokenami i sumą długości B, gdy dodawane U jest jednostajne na {0,…,8}, E[U]=4 i oczekiwany procentowy narzut wynosi **100·4T/B**. Podano średnią tego wyrażenia po 30 zapisanych śladach. Ta zmiana wymaga jawnego oznaczenia innego rodzaju wielkości. Narzuty deterministycznych obron odtwarzają się dokładnie.

## Zależność pięciu wariantów scenariusza

Dodano oddzielną analizę wrażliwości, która uśrednia pięć wariantów każdego z **60 scenariuszy**, następnie losuje scenariusze do bootstrapu i wykonuje sparowane testy Wilcoxona na średnich scenariuszy. Zachowuje ona punktową średnią głównego korpusu, lecz zmienia jednostkę testu. Nie jest nową próbą niezależnych modeli ani naprawą pomiaru.

| Model | Średnia φ | 95% CI, bootstrap scenariuszy |
|---|---:|---:|
| Qwen2.5-1.5B | 0,568040 | [0,551951; 0,583434] |
| Qwen2.5-3B | 0,552073 | [0,539275; 0,565162] |
| Llama-3.2-3B | 0,536020 | [0,520480; 0,551147] |
| Gemma-2-2B | 0,517013 | [0,507586; 0,526610] |
| Phi-1.5 | 0,538256 | [0,524818; 0,551625] |
| TinyLlama — diagnostyka | 0,030199 | [0,026248; 0,034498] |
| Phi-3.5-mini — diagnostyka | 0,043399 | [0,038387; 0,048944] |

Nadal **te same 17/21 par** przekracza próg istotności po korekcji Holma. Przedziały są szersze dla sześciu modeli, maksymalnie o 37,7% dla Qwen2.5-3B; dla Phi-1.5 przedział jest o 1,3% węższy. Brak zmiany decyzji nie dowodzi niezależności wariantów. MAP ma 30 różnych scenariuszy w zbiorze testowym, więc jego krzyżowy bootstrap po promptach i ziarnach nie powiela pięciu wariantów tego samego scenariusza w teście; pozostaje jednak współdzielenie scenariuszy z kalibracją.

Przykładowo Qwen2.5-1.5B minus Qwen2.5-3B ma średnią różnicę 0,015968, scenariuszowe r rangowo-dwuseryjne 0,3235 i p po Holm 0,1184. Duże kontrasty zawierające wadliwe referencje nie stanowią dowodu dużej różnicy podatności. Nie utożsamia się przedziału dla średniej z testem rang ani nie traktuje niekorygowanego przedziału jako odpowiednika testu po Holm.

## Kontrola błędów interpretacji — 11/11

| Typ | Ocena w tym materiale |
|---|---|
| Paradoks Simpsona | Pary mają identyczne prompty i równe wagi tematów; całkowite odwrócenie wszystkich kierunków wewnątrz tematów względem średniej jest przy tych wagach niemożliwe. Lokalne różnice tematów nie dowodzą ogólnego mechanizmu. |
| Błąd ekologiczny | Średnie modeli nie określają poprawności pojedynczej rekonstrukcji; przykłady nie estymują częstości błędów. |
| Paradoks Berksona | Dobór syntetycznego korpusu i podzbiorów EOS ogranicza populację wnioskowania; nie wykazano reprezentatywności produkcji. |
| Błąd kolidera | Długość odpowiedzi i truncation są wielkościami po generacji; opisowa regresja nie identyfikuje przyczynowego wpływu modelu. |
| Pomijanie częstości bazowej | LR@0,5 to próg leksykalny, a nie czułość, PPV czy prawdopodobieństwo ujawnienia. Takiej kalibracji nie wykonano. |
| Regresja do średniej | Brak badania poprawy wybranych skrajnych uczestników; powtórzenia ziaren dotyczą tych samych promptów i nie są nowymi niezależnymi promptami. |
| Błąd przeżywalności | Główna tabela zachowuje wszystkie 300 prób, także obcięte; EOS-only pozostaje osobnym selekcyjnym opisem. |
| Efekt wielu poszukiwań | Sprawdzono wszystkie 21 par i 10 porównań obron, z korekcją w jawnych rodzinach. Tematy i nowa analiza scenariuszowa są eksploracyjne. |
| Wiele ścieżek analizy | Nie poświadczono historycznej prerejestracji ani wszystkich ziaren; nie przedstawia się doboru obecnych analiz jako uprzednio potwierdzającego. |
| Korelacja a przyczynowość | Cross-model i regresja są warunkowe względem aparatury. Zmiana wyniku po transformacji dotyczy całego pipeline’u, w tym segmentacji i obcinania wejścia. |
| Odwrócona przyczynowość | Kierunek wykonania transformacji przed rekonstrukcją jest określony kodem; korelacje cech odpowiedzi z φ nie mają udowodnionego kierunku przyczynowego. |

## Odtworzenie i pliki

```bash
PYTHONHASHSEED=0 python3 experiment_validation/revalidation_20261031/statistics/recompute_all_statistics.py
python3 -m unittest experiment_validation.tests.test_statistics_numerics -v
```

Python 3.10.12, NumPy 1.22.0, SciPy 1.8.0, 10 000 losowań bootstrapu; cztery celowane testy napraw kodu przeszły. Drobne różnice rzędu dokładności zmiennoprzecinkowej mogą zależeć od bibliotek. Historyczna kopia kodu znajduje się w `source_before/`; ponowne obliczenie po naprawie ogonów p celowo daje prawidłowe małe wartości zamiast zer z archiwum.

- [Kontrola wszystkich metryk](metric_checks.json).
- [Porównanie agregatów z archiwum](aggregate_comparisons.json).
- [Niezależne testy główne](independent_primary_pairwise.json) i [obron](independent_defense_pairwise.json).
- [Analiza scenariuszowa](scenario_cluster_sensitivity.json).
- [Wyniki odtworzenia starych agregatów](analysis_rerun/).
- [Agregaty po naprawach kodu](source_fixed_rerun/) i [walidacja napraw](source_fix_validation.json).
- [Oczekiwany narzut Qwen](qwen_1_5b_random_padding_expected_overhead.json) i [Llama](llama_3_2_3b_random_padding_expected_overhead.json).

Oryginalnych wyników w `analysis/` nie nadpisano. Raport nie poświadcza poprawnych wejść T5, detokenizacji ani wartości informacyjnej sygnału; te kwestie wymagają odrębnej walidacji i, w razie potrzeby, nowych rekonstrukcji lub generacji.
