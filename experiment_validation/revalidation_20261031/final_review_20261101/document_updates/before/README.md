# Walidacja eksperymentalna LLeakM

Ten katalog zawiera pipeline zbierania danych, rekonstrukcji, analizy statystycznej i oceny obron dla artykułu o odtwarzaniu odpowiedzi LLM z sekwencji długości tokenów w kontrolowanym streamingu SSE.

## Zakres badania

W historycznym eksperymencie zamierzonym sygnałem była długość bajtowa osobno dekodowanego tokenu w zdarzeniu strumienia aplikacyjnego. Eksperyment nie obejmuje ekstrakcji ramek z zaszyfrowanego ruchu TLS/HTTP2/QUIC. Modele są oceniane zamrożonym rekonstruktorem Weiss-style T5, ale walidacja z 31 października 2026 r. ujawniła różnicę między zamierzonym sygnałem a zarchiwizowaną implementacją kolektora: `iter_lines()` usuwało znaki końca linii wewnątrz tokenów. Replay tej implementacji na zachowanych logach zmienia 2073/2100 śladów i liczbę segmentów w 764 próbkach. Nie odtwarza to dowodu historycznych wejść T5, których dokładnych wektorów nie zapisano.

Poniższe statystyki głównego zbioru opisują zachowane wyniki, z zastrzeżeniem nierozstrzygniętego pochodzenia obserwacji. TinyLlama i Phi-3.5-mini są traktowane diagnostycznie: odpowiednio 299/300 i 162/300 referencji nie zawiera spacji ASCII, a zgodności osobnego dekodowania tokenów z dekodowaniem całej sekwencji nie potwierdzono. Ich niskich wyników nie można przypisać wyłącznie transferowi rekonstruktora. Pilotaże i MAP pobierają `token_utf8_len` bezpośrednio z logów; nie korzystają z tego samego kolektora co główna kampania.

Po audycie naprawiono kod dla nowych przebiegów. `run_validation.py` korzysta z jawnego oracle'a długości z logu (`validated-log-payload-byte-lengths-v1`), sprawdza sumę odebranych bajtów i zapisuje dokładny wektor wejściowy wraz z hashami. Jest to obserwacja instrumentowana. `app.py` tworzy fragmenty z uwzględnieniem kontekstu dekodowania, weryfikuje ich zgodność z pełnym tekstem i zapisuje token IDs oraz pochodzenie. Nie naprawia to starych referencji ani nie poświadcza wcześniejszych wejść T5. Nowe rekonstrukcje są przechowywane osobno w `revalidation_20261031/inference/`; ich kompletność określa `status.json`, a częściowych przebiegów nie należy używać jako zamiennika pełnej tabeli.

Główny zbiór obejmuje:

- 300 wyrównanych promptów na model,
- 15 tematów po 20 promptów,
- 7 modeli ofiary i 2100 próbek,
- `max_new_tokens=512`, `temperature=0`, `top_p=1`,
- wszystkie dostępne segmenty (`max_sentences=0`), 3 próbki rekonstruktora na segment i 3 kandydatów pierwszego segmentu,
- seed analiz `20260706`.

Główna kampania to `full_20260920`. Oddzielne pilotaże kontroli i obron zachowują budżet 96 tokenów i maksymalnie trzy segmenty. Zapisane badanie MAP jest kompletne (22 ustawienia, 3300 rekonstrukcji), ale używa śladów z budżetem 512 tokenów i wyłącznie pierwszego segmentu, ocenianego względem całej referencji. Nie jest to test odporności rekonstrukcji wszystkich segmentów. We wszystkich protokołach wejście i wyjście T5 mają limit 80 tokenów T5; budżet ofiary jest odrębnym parametrem. Transformacja śladu poprzedza segmentację, więc poszczególne warunki nie muszą przekazywać dekoderowi tego samego fragmentu obserwacji.

Phi-1.5 jest modelem bazowym i pełni rolę stress testu. TinyLlama i Phi-3.5-mini pozostają diagnostyką aparatury do czasu walidacji referencji; także interpretacja pozostałych serii jest warunkowa wobec nieustalonego pochodzenia historycznych wejść.

## Struktura

```text
experiment_validation/
├── prompts/            # 15 plików tematycznych, łącznie 300 promptów
├── results/            # próbki, konfiguracje i podsumowania przebiegów
├── scripts/            # zbieranie, statystyki, baseline'y, obrony, manifest
├── analysis/           # historyczne metryki, testy, tabele i figury
├── revalidation_20261031/ # audyty, przeliczone statystyki i osobne nowe przebiegi
└── models.json         # pomocnicza lista dla run_model_matrix.py
```

Tematy: `beliefs_values`, `education`, `employment`, `family_planning`, `financial_status`, `legal_issues`, `mental_health`, `personal_health`, `personal_identity`, `physical_health`, `relationships`, `safety_security`, `self_esteem`, `sexual_health` i `substance_use`.

## Przebiegi użyte w artykule

| Model | Katalog przebiegu | N | Średnie `phi` | `LR@0.5` |
|---|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | `run_full_20260920_qwen_1_5b_full` | 300 | 0.5680 | 76.33% |
| Qwen2.5-3B-Instruct | `run_full_20260920_qwen_3b_full` | 300 | 0.5521 | 70.67% |
| Phi-1.5 | `run_full_20260920_phi_1_5_full` | 300 | 0.5383 | 65.33% |
| Llama-3.2-3B-Instruct | `run_full_20260920_llama_3_2_3b_full` | 300 | 0.5360 | 68.33% |
| Gemma-2-2B-it | `run_full_20260920_gemma_2_2b_full` | 300 | 0.5170 | 59.67% |
| TinyLlama-1.1B-Chat (diagnostyka) | `run_full_20260920_tinyllama_1_1b_full` | 300 | 0.0302 | 0.00% |
| Phi-3.5-mini-instruct (diagnostyka) | `run_full_20260920_phi_3_5_mini_full` | 300 | 0.0434 | 0.00% |

Źródłem tej tabeli jest `analysis/model_stats.json`, który wskazuje dokładne katalogi wejściowe. `analysis/repro_manifest.json` pozostaje historycznym artefaktem z nieaktualnymi wpisami protokołu i hashy; nie jest aktualnym poświadczeniem kampanii. Osobny `analysis/revision_snapshot_20261031.json` zapisuje stan plików podczas rewizji, bez przypisywania go historycznemu wykonaniu. Starsze przebiegi 96-tokenowe nadal są wejściem opisanych niżej pilotaży.

## Metryki

Finalna analiza porównawcza (`scripts/analyze_stats.py`) przelicza dla każdej próbki jeden stały cosinus częstości słów `phi`. Jest to miara pokrycia leksykalnego, a nie Sentence-BERT, BERTScore ani inna miara semantyczna.

Raportowane są:

- średnie `phi` z bootstrapowym 95% CI,
- `LR@0.5`, czyli udział próbek z `phi > 0.5`, z przedziałem Wilsona,
- znormalizowana odległość Levenshteina `ed_norm`,
- precision ROUGE-1 i ROUGE-L,
- sparowane testy Wilcoxona z korektą Holma i rank-biserial effect size (17 z 21 porównań w zapisanych wynikach spełnia próg 0,05 po korekcji),
- kowariaty długości odpowiedzi i regresja kontrolująca długość oraz truncation.

Pola `phi_cosine` i progi `ASR_*` zapisane w pierwotnych `summary.json` pochodzą z czasu danego uruchomienia i mogą używać pierwotnie wybranego backendu. Źródłem liczb cytowanych w artykule są `analysis/per_prompt_phi.csv`, `analysis/model_stats.json` oraz wygenerowane z nich tabele. Ich zgodność rachunkowa nie potwierdza poprawności wcześniejszej rejestracji sygnału ani referencji.

## Ponowne obliczenie statystyk — 31 października 2026

[Raport walidacji](revalidation_20261031/statistics/WALIDACJA_STATYSTYK.md) obejmuje niezależne przeliczenie 2100 głównych par tekstów i ich pierwszych zdań, 240 rekordów kontroli, 420 rekordów obron oraz 3300 rekordów MAP. Sprawdzone zapisane metryki ED i ROUGE oraz `phi` kontroli, obron i MAP zgadzają się dokładnie; główne `phi` ujednolicono jako kosinus częstości słów. Naprawiono agregację percentyli oraz obliczanie bardzo małych wartości p bez zaokrąglania do zera. Decyzje 17/21 porównań modeli i 10/10 porównań obron pozostają takie same. Oryginalnych agregatów nie nadpisano; poprawione wyniki znajdują się w `statistics/source_fixed_rerun/` względem katalogu rewalidacji.

Eksploracyjna analiza zależności wariantów uśrednia pięć promptów każdego z 60 scenariuszy, po czym wykonuje bootstrap scenariuszy (10 000 losowań) i testy sparowane na ich średnich. Zachowuje te same 17 istotnych par po korekcji Holma. Przedziały są szersze dla sześciu modeli, maksymalnie o 37,7%; dla Phi-1.5 są o 1,3% węższe. Nie dowodzi to niezależności wariantów ani generalizacji na nowe scenariusze.

| Model | Średnie `phi` | 95% CI, bootstrap 60 scenariuszy |
|---|---:|---:|
| Qwen2.5-1.5B | 0.5680 | [0.5520, 0.5834] |
| Qwen2.5-3B | 0.5521 | [0.5393, 0.5652] |
| Phi-1.5 | 0.5383 | [0.5248, 0.5516] |
| Llama-3.2-3B | 0.5360 | [0.5205, 0.5511] |
| Gemma-2-2B | 0.5170 | [0.5076, 0.5266] |
| Phi-3.5-mini (diagnostyka) | 0.0434 | [0.0384, 0.0489] |
| TinyLlama-1.1B (diagnostyka) | 0.0302 | [0.0262, 0.0345] |

Pełne wyniki i efekty par zapisano w [scenario_cluster_sensitivity.json](revalidation_20261031/statistics/scenario_cluster_sensitivity.json). Są warunkowe względem zachowanych tekstów i nie usuwają problemów pomiaru.

Narzut `rand_pad_8` ma niespójne zapisy historyczne i nie odtwarza się z udokumentowanego przepisu ziaren. Artykuł podaje teraz **oczekiwany** narzut payloadu: **79,755339%** dla Qwen i **83,133614%** dla Llama (po zaokrągleniu 79,8% i 83,1%). To średnia `100 × 4T/B` po 30 śladach modelu, ponieważ dodawane paddingi są jednostajne na `{0,…,8}`, a ich wartość oczekiwana wynosi 4. Nie jest to odzyskana realizacja historyczna. Punkt wykresu używa średniego oczekiwanego kosztu 81,4% oraz archiwalnej redukcji wyniku; deterministyczne narzuty pozostają bez zmian.

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

Aktualne prace walidacyjne są odrębne od tego archiwum: `revalidation_20261031/statistics/` zawiera przeliczenia statystyk, `collection/` zweryfikowane wektory długości, `references/` audyt detokenizacji, a `inference/` protokoły, próbki i stan nowych rekonstrukcji. Pełna walidacja eksperymentu wymaga kompletnego przebiegu oraz poprawnych referencji; samo przeliczenie zapisanych tekstów jej nie zastępuje.

Ograniczenia interpretacyjne i pełne wyniki opisuje `../report_LLeakM.md` oraz manuskrypt `../article1/template.tex`. Publiczne repozytorium: https://github.com/jackaugustyn/LLeakM
