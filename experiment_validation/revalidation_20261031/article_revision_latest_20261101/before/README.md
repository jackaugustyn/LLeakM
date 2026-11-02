# Walidacja eksperymentalna LLeakM

Pełne ponowne obliczenia zakończono **1 listopada 2026 o 16:15 czasu polskiego**. Wykonano i zweryfikowano **6600 rekonstrukcji T5** oraz **600 nowych generacji modeli ofiar**. Wszystkie 12 etapów kolejki zakończyło się powodzeniem; kompletność potwierdzają niezależne kontrole zapisanych wejść, tekstów i metryk.

## Bieżące wyniki i ich zakres

| Zakres | Liczba | Wynik kontroli |
|---|---:|---|
| Rekonstrukcje z poprawnych długości zapisanych w logach, siedem modeli | 2100 | Kompletne; dwie archiwalne serie referencji pozostają diagnostyczne |
| Kontrole śladu w pilotażu 96-tokenowym | 240 | Kompletne |
| Obrony w pilotażu 96-tokenowym | 360 | Kompletne; zapisano także rzeczywiste losowania paddingu |
| MAP, 512-tokenowe referencje, pierwszy segment, pięć ziaren | 3300 | Kompletne |
| Nowe odpowiedzi TinyLlama i Phi-3.5 | 600 | Token IDs i pełne dekodowanie zweryfikowane |
| Rekonstrukcje nowych odpowiedzi TinyLlama i Phi-3.5 | 600 | Kompletne |

[Raport obliczeń](revalidation_20261031/WYNIKI_WZNOWIENIA.md), [stan ukończonej kolejki](revalidation_20261031/queue/status.json) i [wyniki maszynowe](revalidation_20261031/completed_results.json) opisują skończone przebiegi. `completed_manifest.json` wiąże artefakty w chwili zakończenia kolejki, przed późniejszą integracją manuskryptu; nie jest poświadczeniem historycznej kampanii.

## Rekonstrukcje na archiwalnych referencjach

Poniższe wartości pochodzą z nowych rekonstrukcji T5 na dokładnych długościach payloadów z logów. Nie są średnimi starych predykcji. Każdy model ma 300 tych samych promptów. Referencje TinyLlama i Phi-3.5 w tej tabeli nadal pochodzą z wadliwego, niezależnego dekodowania tokenów.

| Model | N | Średnia φ | LR@0,5 | 95% CI, bootstrap 60 scenariuszy |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 300 | 0.5355 | 65.33% | [0.5195, 0.5511] |
| Qwen2.5-3B-Instruct | 300 | 0.5251 | 63.00% | [0.5119, 0.5385] |
| Llama-3.2-3B-Instruct | 300 | 0.5146 | 61.00% | [0.4981, 0.5312] |
| Gemma-2-2B-it | 300 | 0.4820 | 42.33% | [0.4716, 0.4928] |
| Phi-1.5 | 300 | 0.5164 | 56.67% | [0.5019, 0.5306] |
| TinyLlama-1.1B-Chat — diagnostyka | 300 | 0.0792 | 0.00% | [0.0739, 0.0849] |
| Phi-3.5-mini-instruct — diagnostyka | 300 | 0.1210 | 0.00% | [0.1144, 0.1280] |

Źródło: [pełne nowe statystyki](revalidation_20261031/inference_validation/cuda_full/statistics.json). Po korekcji Holma istotnych jest **15/21 par** zarówno przy jednostce promptu, jak i przy średnich 60 scenariuszy; historyczne 17/21 nie opisuje tych nowych predykcji. Miary pozostają leksykalne i nie określają prawdopodobieństwa ujawnienia danych.

## Nowe referencje — odrębny eksperyment

| Model | N | Średnia φ | LR@0,5 | EOS |
|---|---:|---:|---:|---:|
| TinyLlama-1.1B-Chat | 300 | 0.4352 | 30.33% | 177/300 |
| Phi-3.5-mini-instruct | 300 | 0.5042 | 52.33% | 0/300 |

Wszystkie 600 nowych referencji zawiera spacje i zgadza się z pełnym dekodowaniem zapisanych token IDs. Dodatkowy audyt stałych sekwencji IDs wykazał, że stary dekoder pojedynczych tokenów odtwarza dokładnie wszystkie 600 archiwalnych tekstów, fragmentów i wektorów długości; pełne dekodowanie tych samych nowych IDs daje poprawne granice słów. Potwierdza to mechanizm błędu detokenizacji dla tych sekwencji, bez odzyskania niearchiwizowanych historycznych IDs. [Dowód i zakres](revalidation_20261031/final_review_20261101/fresh_legacy_replay.md). Ofiary działają w FP16; Phi-3.5 korzysta częściowo z RAM, bez kwantyzacji. Rewizje wag, format promptu `legacy`, limit 512 tokenów, ziarna i mapa urządzeń są zapisane w protokołach. Nowych odpowiedzi nie łączy się z pięcioma historycznymi seriami w jeden zweryfikowany ranking. Dla 600 dopasowanych przypadków potwierdzono zgodność starego renderowania z archiwalnymi tekstami i wektorami oraz identyczne ziarna, wagi, kod i parametry T5. Porównanie zmienia równocześnie wejściowy ślad długości i tekst referencyjny; nie rozdziela ich wpływu na wynik. Nie odzyskano historycznych IDs ani wersji wag ofiar.

Szczegóły: [TinyLlama](revalidation_20261031/inference_validation/fresh_tinyllama_cuda/statistics.json), [Phi-3.5](revalidation_20261031/inference_validation/fresh_phi_cuda/statistics.json), [protokoły nowych referencji](revalidation_20261031/fresh_references/PRZYGOTOWANIE.md).

## Kontrole i obrony po ponownej inferencji

[Nowe wyniki 42 komórek](revalidation_20261031/secondary/cuda_full/summary.json) zastępują stare liczby w bieżącym artykule. Wszystkie 3300 predykcji MAP odtworzono dokładnie względem archiwum; zmiany niektórych przedziałów wynikają z jawnie zapisanego nowego przepisu bootstrapu, a nie ze zmiany tych predykcji. W pilotażach użyto jawnych nowych ziaren BLAKE2s; MAP zachowuje wcześniejsze ziarna, podział 30/270 i kalibrację. Zapisy zawierają dokładne wektory przed i po transformacji, wejścia po inwersji, rzeczywiste tokeny enkodera, wybrane teksty kandydatów i koszty payloadu.

Prawdziwy ślad nie przewyższa każdego wariantu kontrolnego: w sześciu kontrastach kontrolnych po Holm istotne są trzy. Dla Llama średnia `cross_prompt` jest nieco wyższa niż `real`. Nie należy więc zachowywać dawnego ogólnego wniosku o przewadze prawdziwego śladu nad wszystkimi kontrolami. Koszty nowego losowego paddingu są zapisanymi realizacjami; wcześniejsze 79,8%/83,1% były wartościami oczekiwanymi dla nieodtwarzalnych historycznych losowań.

## Pomiar, metryki i ograniczenia

`run_validation.py` używa jawnego oracle'a długości z logu (`validated-log-payload-byte-lengths-v1`), sprawdza sumę odebranych bajtów i zapisuje wejściowy wektor wraz z hashami. `app.py` tworzy fragmenty z kontekstem dekodowania i sprawdza zgodność z pełną sekwencją. To instrumentowany eksperyment długości payloadu, bez wykazanego odtwarzania granic zdarzeń z TLS/HTTP/2/QUIC.

Wyniki obejmują kosinus częstości słów φ, odległość Levenshteina podzieloną przez długość referencji, ROUGE-1 i ROUGE-L, częstość przekroczenia progu leksykalnego oraz przedziały i testy sparowane. Główne i nowe referencje mają po 300 promptów/60 scenariuszy. Limity enkodera i dekodera T5 wynoszą 80 tokenów. Główny przebieg żąda wszystkich segmentów i trzech kandydatów; pilotaże najwyżej trzech segmentów i trzech kandydatów; MAP pierwszego segmentu i jednego kandydata na ziarno. Cały zrekonstruowany tekst jest oceniany względem całej referencji. Zmiana transformacji zmienia też segmentację i pokrycie wejścia.

## Zachowane archiwum

Pierwotne `results/` i `analysis/` pozostają niezmienionym źródłem historycznym. [Rewalidacja statystyk archiwalnych](revalidation_20261031/statistics/WALIDACJA_STATYSTYK.md) potwierdziła ich arytmetykę, ale sama nie naprawiała pomiaru. Dawne wyniki 0,030–0,568 i 17/21 istotnych par dotyczą tamtych predykcji; nowe wyniki znajdują się w `revalidation_20261031/`.

`analysis/repro_manifest.json` ma historyczne, nieaktualne wpisy. `analysis/revision_snapshot_20261031.json` opisuje wcześniejszą rewizję tekstu. Nowe protokoły i manifest zakończonej kolejki nie poświadczają tożsamości wcześniejszych wag, kodu ani wejść rekonstruktora.

## Struktura

- `results/`, `analysis/`, `../logs/`: dane i analizy historyczne.
- `revalidation_20261031/inference/`: nowe rekonstrukcje, wejścia enkodera i protokoły.
- `revalidation_20261031/inference_validation/`: niezależne kontrole i kompletne statystyki.
- `revalidation_20261031/secondary/cuda_full/`: 3900 nowych prób kontroli/obron/MAP.
- `revalidation_20261031/fresh_references/`: 600 nowych generacji i ich logi.
- `revalidation_20261031/final_review_20261101/`: końcowy audyt i integracja artykułu.

Poniższe polecenia opisują ogólne narzędzia oraz zachowane analizy historyczne. Nie nadpisują automatycznie tabel artykułu wynikami zakończonej kolejki.

## Uruchomienie pojedynczego modelu

Najpierw uruchom serwer z wybranym modelem ofiary:

```bash
. .venv/bin/activate
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct HF_HUB_OFFLINE=1 \
  python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

W drugim terminalu poniższe parametry odpowiadają budżetom głównej kampanii. Naprawiony kolektor i harness stosują nowy kontrakt obserwacji i referencji, więc wyniki nowego przebiegu należy zachować oddzielnie od historycznych danych. Walidacja logu i referencji nie oznacza odtworzenia historycznych odpowiedzi:

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_validation.py \
  --max-prompts 300 \
  --max-new-tokens 512 \
  --temperature 0 \
  --top-p 1 \
  --samples-per-segment 3 \
  --max-sentences 0 \
  --num-first-candidates 3 \
  --prompt-format legacy \
  --reconstruction-seed 20260706 \
  --label qwen_1_5b_full
```

Usuń `HF_HUB_OFFLINE=1`, jeśli modele nie znajdują się jeszcze w lokalnym cache Hugging Face. Szybki smoke test można wykonać przez `--max-prompts 20`, ale jego wynik nie jest porównywalny z finalnym benchmarkiem.

Każdy przebieg tworzy `results/run_YYYYMMDD_HHMMSS_<label>/` z plikami:

- `config.json`,
- `progress.json`,
- `samples.jsonl`,
- `summary.json`,
- `summary.md`,
- `summary_by_topic.csv`.

## Macierz modeli

`scripts/run_model_matrix.py` kolejno uruchamia wskazaną listę modeli. Jego domyślne wartości historyczne (`72` tokeny i `2` próbki na segment) różnią się od protokołu artykułu, dlatego parametry trzeba podać jawnie:

```bash
. .venv/bin/activate
python experiment_validation/scripts/run_model_matrix.py \
  --models-file experiment_validation/models_publication.json \
  --prompts 300 \
  --max-new-tokens 512 \
  --samples-per-segment 3 \
  --max-sentences 0 \
  --num-first-candidates 3 \
  --prompt-format legacy \
  --reconstruction-seed 20260706 \
  --hf-offline
```

Opcja `--clear-results` usuwa dotychczasową zawartość katalogu wyników i nie jest potrzebna do zwykłego uruchomienia. `models.json` jest listą pomocniczą; siedem katalogów użytych w głównej tabeli wskazuje `analysis/model_stats.json` oraz tabela powyżej.

## Odtworzenie analiz

```bash
. .venv/bin/activate
PYTHONHASHSEED=0 python experiment_validation/scripts/analyze_stats.py
PYTHONHASHSEED=0 python experiment_validation/scripts/analyze_extras.py
python experiment_validation/scripts/make_figures.py
```

Wyniki trafiają do `analysis/` i zastępują istniejące wygenerowane pliki; porównanie bez nadpisania wymaga osobnej kopii roboczej. `analyze_stats.py` tworzy główne statystyki i testy, `analyze_extras.py` generuje kontrole długości, regresję, dane jakościowe i tabele pomocnicze, a `make_figures.py` generuje fragmenty TikZ/PGFPlots. Obecny `article1/template.tex` zawiera tabele i wykresy bezpośrednio, więc te skrypty nie aktualizują automatycznie jego treści. Skrypt `make_repro_manifest.py` nadpisuje historyczny manifest i opis środowiska; nie należy używać takiego nadpisania jako dowodu pochodzenia wcześniejszych wyników.

## Baseline'y śladu i obrony

Pilotaże używają po 30 promptów (po dwa na temat) dla Qwen2.5-1.5B i Llama-3.2-3B ze starszych przebiegów 96-tokenowych `run_20260325_104518_qwen_1_5b` i `run_20260414_073030_llama_3_2_3b`. Długości pobierane są z `token_utf8_len` w logach. Rekonstrukcja obejmuje maksymalnie trzy segmenty, a wyniki odnoszą się do całej zapisanej referencji.

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 python experiment_validation/scripts/trace_baselines.py \
  --per-topic 2 --samples-per-segment 3 --max-sentences 3

HF_HUB_OFFLINE=1 PYTHONHASHSEED=0 python experiment_validation/scripts/defense_eval.py \
  --per-topic 2 --samples-per-segment 3 --max-sentences 3
```

Kontrole porównują ślad rzeczywisty z sekwencją przetasowaną, stałą i pochodzącą z innego promptu. Są to częściowe ablacje: przetasowanie zachowuje histogram i liczbę elementów, a stały ślad zachowuje liczbę elementów i zaokrągloną średnią długość. Nie stanowią kontroli całkowicie pozbawionej informacji. Obrony obejmują `bucket_8`, `pad_32`, `batch_2`, `batch_4` i `rand_pad_8`; ocena dotyczy zamrożonego, nieadaptacyjnego rekonstruktora. Receptura seedów zależy od Pythonowego `hash()` i ustawienia `PYTHONHASHSEED`. Polecenia powyżej ustalają je dla nowego uruchomienia, ale nie potwierdzają historycznego środowiska ani niezapisanych seedów per próbka.

## Wzmocniony protokół: wspólne seedy i adaptacyjny atakujący

`scripts/robustness_eval.py` odtwarza zapisane ślady pięć razy, z tym samym
seedem dekodera dla wszystkich warunków danego promptu. Domyślnie używa po dwa
prompty z każdego z 15 tematów jako zbiór testowy (30 na model), a pozostałe
270 jako rozłączny zbiór kalibracyjny. Adaptacyjny atakujący uczy się na nim
empirycznego odwzorowania MAP ze śladu po obronie do długości czystych; dla
batchingu uczy się także rozkładów krotek o danej sumie. Nie jest to ponowne
trenowanie T5 i tak właśnie należy go opisywać w artykule.

Zapisany protokół `028a07f6eb2ebdd2` używa parametrów `--samples-per-segment 1 --num-first-candidates 1 --max-sentences 1`, a nie domyślnych trzech próbek i wszystkich segmentów. Najpierw można sprawdzić odpowiadający mu plan bez ładowania modeli:

```bash
python experiment_validation/scripts/robustness_eval.py \
  --samples-per-segment 1 --num-first-candidates 1 --max-sentences 1 --plan
```

Przebieg dwóch modeli z parametrami zapisanego badania:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python experiment_validation/scripts/robustness_eval.py \
  --samples-per-segment 1 --num-first-candidates 1 --max-sentences 1
```

Domyślne `--max-sentences 0` oznacza rekonstrukcję wszystkich dostępnych segmentów i definiowałoby inny eksperyment niż zapisane wyniki. Dla pierwszego segmentu wynik liczony jest względem całej odpowiedzi. Kalibracja i test są rozłączne po dokładnym prompcie, lecz nie po scenariuszu: korpus zawiera 60 scenariuszy po pięć wariantów, a 120 z 270 promptów kalibracyjnych dzieli scenariusze z 30 promptami testowymi. Nie wykazano generalizacji na nowe scenariusze. Powstają `analysis/robustness_samples.jsonl`, konfiguracja z
identyfikatorem protokołu oraz `analysis/robustness_eval.json`. Tabela LaTeX
jest generowana dopiero wtedy, gdy wszystkie oczekiwane komórki są kompletne.
Wznowienie jest bezpieczne, ponieważ klucz obejmuje model, prompt, obronę,
wariant atakującego i seed. Opcja `--overwrite` jest celowa i usuwa wyłącznie
wskazany plik wynikowy.

## Pełne odpowiedzi

Główna kampania z budżetem 512 tokenów została już wykonana; nie wszystkie jej odpowiedzi osiągają EOS. `app.py` zapisuje `finish_reason` oraz
`response_complete`; token EOS nie jest włączany do tekstu ani śladu. Dla
porównywalności z pierwotną kampanią można zachować format promptu `legacy`:

```bash
python experiment_validation/scripts/run_model_matrix.py \
  --models-file experiment_validation/models_publication.json \
  --prompts 300 \
  --max-new-tokens 512 \
  --samples-per-segment 3 \
  --max-sentences 0 \
  --num-first-candidates 3 \
  --prompt-format legacy \
  --reconstruction-seed 20260706 \
  --hf-offline
```

Wyniki pełnych odpowiedzi należy analizować z filtrem `--complete-only`; skrypt
przerwie pracę, jeśli w którymkolwiek temacie zabraknie wymaganej liczby
odpowiedzi zakończonych EOS. Katalogi nowych przebiegów podaje się jawnie, np.:

```bash
python experiment_validation/scripts/robustness_eval.py \
  --run-spec qwen_full=run_YYYYMMDD_HHMMSS_qwen_1_5b_full \
  --run-spec llama_full=run_YYYYMMDD_HHMMSS_llama_3_2_3b_full \
  --complete-only
```

Nie należy nazywać odpowiedzi „pełną” wyłącznie dlatego, że zwiększono limit:
do analizy complete-only kwalifikuje ją dopiero `finish_reason=eos`.

## Reprodukowalność

- `analysis/REPRODUCIBILITY.md` — historyczny opis środowiska, protokołu i poleceń; może zawierać nieaktualne parametry,
- `analysis/repro_manifest.json` — historyczny manifest, z nieaktualnymi wpisami; zachowany bez nadpisania,
- `analysis/revision_snapshot_20261031.json` — snapshot wcześniejszej rewizji tekstu, sprzed późniejszych napraw kodu; bez poświadczenia historycznego wykonania,
- `analysis/model_stats.json` — finalne wyniki modeli,
- `analysis/pairwise_tests.json` — 21 porównań zapisanych wyników, 17 istotnych po korekcji Holma,
- `analysis/defense_eval.json` — wyniki obron,
- `analysis/trace_baselines.json` — baseline'y informacji w śladzie.

Zakończone przebiegi rewalidacji są odrębne od tego archiwum: `revalidation_20261031/statistics/` zawiera przeliczenia statystyk, `collection/` zweryfikowane wektory długości, `references/` audyt detokenizacji, a `inference/` protokoły, próbki i stan nowych rekonstrukcji. Kompletność nowych przebiegów jest potwierdzona. W siedmiomodelowej analizie zachowanych referencji dwie serie nadal pozostają diagnostyczne; oddzielny eksperyment nowych generacji ma zweryfikowane token IDs.

Ograniczenia interpretacyjne i pełne wyniki opisuje `../report_LLeakM.md` oraz manuskrypt `../article1/template.tex`. Publiczne repozytorium: https://github.com/jackaugustyn/LLeakM
