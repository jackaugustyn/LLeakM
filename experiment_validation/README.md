# Experiment Validation

Ten katalog zawiera pełny pipeline walidacji rekonstrukcji metodą Weiss na kilkuset promptach tematycznych.

## Zawartość

- `prompts/*.txt` - osobne pliki tematów (1 prompt na linię)
- `scripts/generate_prompts.py` - generator zestawów promptów (300 promptów / 15 tematów)
- `scripts/run_validation.py` - runner eksperymentu + metryki + raporty
- `results/` - artefakty uruchomień

## Tematyka promptów

Tematy odpowiadają kategoriom ryzyka i poufności omawianym w paperze (m.in. zdrowie, zdrowie seksualne, używki, kwestie prawne, finanse, relacje, bezpieczeństwo):

- `mental_health`
- `physical_health`
- `sexual_health`
- `substance_use`
- `legal_issues`
- `financial_status`
- `employment`
- `education`
- `relationships`
- `family_planning`
- `personal_identity`
- `self_esteem`
- `safety_security`
- `beliefs_values`
- `personal_health`

## Metryki (analogiczne do artykułu)

Dla każdej próbki liczone są:

- `ed_norm` - znormalizowany edit distance (char-level)
- `rouge1_precision`, `rouge1_recall`, `rouge1_f1`
- `rougeL_precision`, `rougeL_recall`, `rougeL_f1`
- `phi_cosine` - podobieństwo tematyczne (cosine)
- `ASR` - odsetek próbek z `phi_cosine > 0.5`

Progi raportowane globalnie:

- `ED=0`, `ED<=0.1`
- `R1=1.0`, `R1>=0.9`
- `RL=1.0`, `RL>=0.9`
- `phi=1.0`, `phi>0.9`, `ASR(phi>0.5)`

## Uruchomienie

1. Start serwera (w osobnym terminalu):

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 uvicorn app:app --host 127.0.0.1 --port 8000
```

2. (Opcjonalnie) regeneracja promptów:

```bash
. .venv/bin/activate
python experiment_validation/scripts/generate_prompts.py
```

3. Walidacja pełna (300 promptów):

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_validation.py \
  --max-prompts 300 \
  --max-new-tokens 96 \
  --samples-per-segment 3 \
  --max-sentences 3
```

4. Szybki test (np. 20 promptów):

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_validation.py --max-prompts 20
```

5. Macierz modeli (np. 2 modele ofiary):

```bash
. .venv/bin/activate
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_model_matrix.py \
  --models-file experiment_validation/models.json \
  --prompts 300 \
  --clear-results \
  --hf-offline
```

`experiment_validation/models.json` domyślnie zawiera:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct` (proponowany drugi model)

## Artefakty wyników

Każde uruchomienie zapisuje nowy folder `results/run_YYYYMMDD_HHMMSS/` z plikami:

- `config.json`
- `progress.json`
- `samples.jsonl` (per prompt: prompt, run_id, ground truth, rekonstrukcja, metryki)
- `summary.json` (global)
- `summary_by_topic.csv` (per temat)
- `summary.md` (czytelny raport)
