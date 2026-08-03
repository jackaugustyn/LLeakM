# Walidacja eksperymentalna LLeakM

Ten katalog zawiera pipeline zbierania danych, rekonstrukcji, analizy statystycznej i oceny obron dla artykułu o odtwarzaniu odpowiedzi LLM z sekwencji długości tokenów w kontrolowanym streamingu SSE.

## Zakres badania

Serwer badawczy emituje dokładnie jeden token w jednej ramce SSE. Eksperyment korzysta więc z idealizowanego oracle'a długości tokenów, a nie z ekstrakcji ramek z zaszyfrowanego ruchu TLS/HTTP2/QUIC. Wszystkie modele ofiary są oceniane tym samym, zamrożonym rekonstruktorem Weiss-style T5.

Główny zbiór obejmuje:

- 300 wyrównanych promptów na model,
- 15 tematów po 20 promptów,
- 7 modeli ofiary i 2100 próbek,
- `max_new_tokens=96`, `temperature=0`, `top_p=1`,
- 3 próbki rekonstruktora na segment, maksymalnie 3 zdania i 3 kandydatów pierwszego zdania,
- seed analiz `20260706`.

Phi-1.5 jest modelem bazowym i pełni rolę stress testu. Główne porównania modeli dostrojonych instrukcyjnie/czatowo obejmują pozostałe sześć modeli.

## Struktura

```text
experiment_validation/
├── prompts/            # 15 plików tematycznych, łącznie 300 promptów
├── results/            # próbki, konfiguracje i podsumowania przebiegów
├── scripts/            # zbieranie, statystyki, baseline'y, obrony, manifest
├── analysis/           # finalne metryki, testy, tabele i figury
└── models.json         # pomocnicza lista dla run_model_matrix.py
```

Tematy: `beliefs_values`, `education`, `employment`, `family_planning`, `financial_status`, `legal_issues`, `mental_health`, `personal_health`, `personal_identity`, `physical_health`, `relationships`, `safety_security`, `self_esteem`, `sexual_health` i `substance_use`.

## Przebiegi użyte w artykule

| Model | Katalog przebiegu | N | Średnie `phi` | `LR@0.5` |
|---|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | `run_20260325_104518_qwen_1_5b` | 300 | 0.3242 | 8.67% |
| Qwen2.5-3B-Instruct | `run_20260327_124817_qwen_3b_final_resume` | 300 | 0.3096 | 3.00% |
| Phi-1.5 | `run_20260329_085714_phi_1_5` | 300 | 0.2747 | 4.33% |
| Llama-3.2-3B-Instruct | `run_20260414_073030_llama_3_2_3b` | 300 | 0.2538 | 3.00% |
| Gemma-2-2B-it | `run_20260415_080034_gemma_2_2b_it` | 300 | 0.1798 | 0.00% |
| TinyLlama-1.1B-Chat | `run_20260325_153010_tinyllama_1_1b` | 300 | 0.0428 | 0.00% |
| Phi-3.5-mini-instruct | `run_20260414_111644_phi_3_5_mini` | 300 | 0.0239 | 0.00% |

Pełna, maszynowo czytelna lista przebiegów i sum SHA-256 znajduje się w `analysis/repro_manifest.json`. Inne katalogi w `results/` są historyczne lub pośrednie i nie stanowią podstawy głównych tez artykułu.

## Metryki

Finalna analiza porównawcza (`scripts/analyze_stats.py`) przelicza dla każdej próbki jeden stały cosinus częstości słów `phi`. Jest to miara pokrycia leksykalnego, a nie Sentence-BERT, BERTScore ani inna miara semantyczna.

Raportowane są:

- średnie `phi` z bootstrapowym 95% CI,
- `LR@0.5`, czyli udział próbek z `phi > 0.5`, z przedziałem Wilsona,
- znormalizowana odległość Levenshteina `ed_norm`,
- precision ROUGE-1 i ROUGE-L,
- sparowane testy Wilcoxona z korektą Holma i rank-biserial effect size,
- kowariaty długości odpowiedzi i regresja kontrolująca długość oraz truncation.

Pola `phi_cosine` i progi `ASR_*` zapisane w pierwotnych `summary.json` pochodzą z czasu danego uruchomienia i mogą używać pierwotnie wybranego backendu. Dla liczb cytowanych w artykule źródłem prawdy są `analysis/per_prompt_phi.csv`, `analysis/model_stats.json` oraz wygenerowane z nich tabele.

## Uruchomienie pojedynczego modelu

Najpierw uruchom serwer z wybranym modelem ofiary:

```bash
. .venv/bin/activate
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct HF_HUB_OFFLINE=1 \
  python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

W drugim terminalu uruchom standardowy przebieg:

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_validation.py \
  --max-prompts 300 \
  --max-new-tokens 96 \
  --temperature 0 \
  --top-p 1 \
  --samples-per-segment 3 \
  --max-sentences 3 \
  --num-first-candidates 3 \
  --label qwen_1_5b
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

`scripts/run_model_matrix.py` kolejno uruchamia modele wymienione w `models.json`. Jego domyślne wartości historyczne (`72` tokeny i `2` próbki na segment) różnią się od protokołu artykułu, dlatego parametry trzeba podać jawnie:

```bash
. .venv/bin/activate
python experiment_validation/scripts/run_model_matrix.py \
  --models-file experiment_validation/models.json \
  --prompts 300 \
  --max-new-tokens 96 \
  --samples-per-segment 3 \
  --max-sentences 3 \
  --num-first-candidates 3 \
  --hf-offline
```

Opcja `--clear-results` usuwa dotychczasową zawartość katalogu wyników i nie jest potrzebna do zwykłego uruchomienia. `models.json` jest listą pomocniczą, natomiast autorytatywna lista siedmiu przebiegów publikacyjnych znajduje się w manifeście.

## Odtworzenie analiz

```bash
. .venv/bin/activate
python experiment_validation/scripts/analyze_stats.py
python experiment_validation/scripts/analyze_extras.py
python experiment_validation/scripts/make_figures.py
python experiment_validation/scripts/make_repro_manifest.py
```

Wyniki trafiają do `analysis/`. `analyze_stats.py` tworzy główne statystyki i testy, `analyze_extras.py` generuje kontrole długości, regresję, dane jakościowe i tabele pomocnicze, a `make_figures.py` generuje fragmenty TikZ/PGFPlots wczytywane przez manuskrypt.

## Baseline'y śladu i obrony

Badanie używa topic-balanced subsetu po 30 promptów dla Qwen2.5-1.5B i Llama-3.2-3B. W repozytorium znajduje się 60 odpowiadających im surowych logów; pozostałe logi runtime są ignorowane.

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 python experiment_validation/scripts/trace_baselines.py \
  --per-topic 2 --samples-per-segment 3 --max-sentences 3

HF_HUB_OFFLINE=1 python experiment_validation/scripts/defense_eval.py \
  --per-topic 2 --samples-per-segment 3 --max-sentences 3
```

Baseline'y porównują ślad rzeczywisty z sekwencją przetasowaną, stałą i pochodzącą z innego promptu. Obrony obejmują `bucket_8`, `pad_32`, `batch_2`, `batch_4` i `rand_pad_8`. Są oceniane przeciwko zamrożonemu, nieadaptacyjnemu rekonstruktorowi.

## Reprodukowalność

- `analysis/REPRODUCIBILITY.md` — środowisko, protokół i polecenia,
- `analysis/repro_manifest.json` — pełne sumy SHA-256,
- `analysis/model_stats.json` — finalne wyniki modeli,
- `analysis/pairwise_tests.json` — 21 porównań modeli,
- `analysis/defense_eval.json` — wyniki obron,
- `analysis/trace_baselines.json` — baseline'y informacji w śladzie.

Ograniczenia interpretacyjne i pełne wyniki opisuje `../report_LLeakM.md` oraz manuskrypt `../article1/tifs_main.tex`.
