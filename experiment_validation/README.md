# Walidacja eksperymentalna LLeakM

**Zakończono pełną kampanię 6600 rekonstrukcji z selektorem ograniczonym do pierwszego EOS.** Wszystkie bieżące wyniki artykułu pochodzą z `selector_eos_20261101/`; integracja tabel, rysunków i tekstu znajduje się w `selector_eos_20261101/article_revision_20261102/`.

Rekonstrukcje wykonano na zamrożonych wejściach i referencjach. W tej kampanii nie generowano nowych odpowiedzi modeli ofiar. Odrębne kohorty TinyLlama i Phi-3.5-mini wykorzystują 600 wcześniej zebranych odpowiedzi z zapisanymi token IDs i zweryfikowanym dekodowaniem całych sekwencji.

## Zakres zakończonych obliczeń

| Eksperyment | Rekonstrukcje | Protokół |
|---|---:|---|
| Siedem modeli, archiwalne referencje | 2100 | 300 promptów na model; wszystkie dostępne segmenty żądane; trzy kandydaty na segment |
| Odrębne zweryfikowane referencje TinyLlama i Phi-3.5-mini | 600 | Po 300 promptów; wszystkie segmenty; trzy kandydaty |
| Kontrole śladu | 240 | Dwa modele, 30 promptów, cztery warunki; do trzech segmentów |
| Pilotaż obron | 360 | Dwa modele, 30 promptów, sześć warunków; do trzech segmentów |
| MAP | 3300 | Dwa modele, 30 promptów, 11 warunków, pięć ziaren; pierwszy segment, jeden kandydat |

[Audyt kompletności i wyboru kandydatów](selector_eos_20261101/validation/audit_campaign.json), [wyniki główne i odrębne referencje](selector_eos_20261101/validation/main_fresh_factledger.json) oraz [audyt kontroli, obron i MAP](selector_eos_20261101/validation/secondary_audit.json) mają status pozytywny. [Manifest zakończonej kampanii](selector_eos_20261101/completed_manifest.json) wiąże zapisane artefakty z ich SHA-256.

## Archiwalne referencje, bieżące rekonstrukcje EOS

| Model | N | Średnia φ | LR@0,5 | 95% CI, bootstrap scenariuszy |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 300 | 0.5768 | 76.00% | [0.5621, 0.5912] |
| Qwen2.5-3B-Instruct | 300 | 0.5571 | 76.67% | [0.5437, 0.5709] |
| Phi-1.5 | 300 | 0.5420 | 65.67% | [0.5270, 0.5570] |
| Llama-3.2-3B-Instruct | 300 | 0.5365 | 68.00% | [0.5206, 0.5522] |
| Gemma-2-2B-it | 300 | 0.5092 | 55.00% | [0.4996, 0.5192] |
| Phi-3.5-mini-instruct (diagnostyka) | 300 | 0.1297 | 0.00% | [0.1231, 0.1367] |
| TinyLlama-1.1B-Chat (diagnostyka) | 300 | 0.0832 | 0.00% | [0.0777, 0.0887] |

Średnie pięciu niediagnostycznych serii wynoszą 0.5092–0.5768. Dwie pozostałe serie odzwierciedlają wadliwy format referencji po niezależnym dekodowaniu tokenów; nie opisują wewnętrznej odporności tych modeli. Przy jednostce promptu oraz przy średnich 60 scenariuszy istotnych jest **18/21 par** po korekcji Holma. Miary są leksykalne, a nie semantyczne.

## Odrębne referencje ze zweryfikowanym dekodowaniem

| Model | N | Średnia φ | LR@0,5 | 95% CI, bootstrap scenariuszy | EOS ofiary |
|---|---:|---:|---:|---|---:|
| TinyLlama-1.1B-Chat | 300 | 0.4677 | 44.33% | [0.4501, 0.4854] | 177/300 |
| Phi-3.5-mini-instruct | 300 | 0.5269 | 62.33% | [0.5127, 0.5411] | 0/300 |

Tych kohort nie łączy się z archiwalnymi seriami w jeden ranking siedmiu modeli. Sparowane porównanie formatów referencji w bieżącej kampanii daje różnice średniej φ równe 0.3845 dla TinyLlama i 0.3972 dla Phi-3.5-mini, z przedziałami scenariuszowymi odpowiednio [0.3662, 0.4029] i [0.3806, 0.4134]. Zmienia ono jednocześnie długości wejściowe i referencje oceny, więc nie rozdziela tych efektów. [Źródło sparowanego kontrastu](selector_eos_20261101/validation/paired_detokenization_score_contrast.json).

## Kontrole i obrony

Po korekcji Holma istotne są **4/6 kontrastów kontrolnych**: ślady przetasowane i stałe obniżają wyniki obu modeli, a oba kontrasty `cross_prompt` nie są istotne. Nie wykazano przewagi rzeczywistego śladu nad każdym wariantem kontrolnym.

W pilotażu zamrożonego rekonstruktora istotnych jest **10/10 kontrastów obronnych**. Redukcje średniej φ wynoszą 43.8–89.6%; koszty to rzeczywiście zapisane narzuty chronionego payloadu. Batching zmniejsza liczbę obserwowanych jednostek bez dodawania bajtów do samego payloadu. Nie zmierzono narzutów transportowych, opóźnień ani rzeczywistego pakietowania.

MAP wykorzystuje empiryczne odwzorowanie długości i wspólne ziarna dekodera. Nie jest ponownym trenowaniem T5. Kalibracja i test są rozłączne po dokładnym prompcie, lecz nie po scenariuszu; nie wykazano generalizacji na nowe scenariusze. Wszystkie jego wartości w artykule pochodzą z 3300 rekordów bieżącej kampanii.

## Selektor i ograniczenia pomiaru

Selektor `eos-through-first-inclusive/1` sumuje przetworzone wyniki przejść do pierwszego EOS włącznie i pomija późniejsze pozycje. Oceniany rekonstruktor zachowuje limit 80 tokenów enkodera i dekodera. Żądanie wszystkich segmentów nie gwarantuje niepustego wyniku każdego segmentu: w części głównej wykonano 62 757 prób segmentów, z których 62 755 zwróciło niepusty tekst. Odrębne zweryfikowane kohorty TinyLlama i Phi-3.5-mini mają odpowiednio 8835 i 10 748 niepustych wyników segmentów. Puste końcowe wyniki dotyczą jednej odpowiedzi Llama i jednej odpowiedzi Phi-1.5.

Wejściem jest instrumentowany oracle długości payloadu z logów. Eksperyment nie wykazuje odtworzenia granic zdarzeń aplikacji z przechwyconego TLS, HTTP/2 ani QUIC. Transformacje zmieniają także segmentację i pokrycie wejścia enkodera. Cały wynik rekonstrukcji jest oceniany względem całej referencji; EOS ofiary oznacza zakończenie generacji, a nie poprawność rekonstrukcji.

## Odtworzenie tabel i rysunków

Poniższe polecenia korzystają z ukończonych audytów i nie uruchamiają modeli:

```bash
python experiment_validation/selector_eos_20261101/article_revision_20261102/regenerate_tables.py
python experiment_validation/selector_eos_20261101/article_revision_20261102/check_generated_tables.py
python experiment_validation/selector_eos_20261101/article_revision_20261102/generate_figures.py
```

Generator tabel zapisuje `tables.json` i `tables.tex`; nie nadpisuje manuskryptu automatycznie. Niezależny parser sprawdza wszystkie 13 tabel: 433 wartości liczbowe i 93 pola opisowe. Generator wykresów zapisuje nowe pliki `article1/figures/eos_model_results.*` i `eos_defense_tradeoff.*`, dokładne dane JSON/CSV oraz manifest źródeł i eksportów. Oba panele wykresu wyników modeli korzystają z bieżącej kampanii EOS; drugi panel porównuje formaty referencji w jej obrębie.

## Artefakty i udostępnienie

- `selector_eos_20261101/inference/`: główne rekonstrukcje i odrębne referencje, protokoły oraz zapis kandydatów.
- `selector_eos_20261101/secondary/cuda_full/`: 3900 prób kontroli, obron i MAP.
- `selector_eos_20261101/validation/`: niezależne audyty, statystyki i kontrast formatów referencji.
- `selector_eos_20261101/article_revision_20261102/`: aktualna integracja artykułu, generatory, tabele, dane wykresów i kontrole.
- `../article1/supplement/LLeakM-S2-EOS-20261102-v1.tar.gz`: lokalny pakiet Supplementary Data S2 z manifestem SHA-256 i weryfikatorem CPU.

Wcześniejsze wyniki i audyty pozostają archiwum. Ich istnienie nie zastępuje nowych dowodów i nie stanowi podstawy bieżących tabel ani wniosków. Zapisane wcześniej referencje i logi są zachowanymi wejściami obecnej kampanii.

Repozytorium projektu: https://github.com/jackaugustyn/LLeakM. Publiczna archiwizacja bieżącego pakietu nadal oczekuje na wykonanie; obecność repozytorium nie potwierdza udostępnienia tej wersji. Zgoda autorów na końcowy PDF i pakiet oraz aktualne oficjalne wymagania czasopisma pozostają odrębnymi pozycjami w [liście zgłoszeniowej](../article1/SUBMISSION_CHECKLIST.md).
