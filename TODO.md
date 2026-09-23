# TODO przed zgłoszeniem artykułu

Stan na 2026-09-23 16:45. Kampania `full_20260920`: **7/7 complete**.
Manuskrypt `article1/template.tex` przepisany na analizę 512 tokenów
(φ 0.568 Qwen2.5-1.5B → 0.030 TinyLlama; 17/21 Holm; tabela EOS).
P3 (pięć seedów) i P4 (adaptacyjny MAP) nadal nieuruchomione — nie wpisywać
ich wyników do abstraktu.

## P0 — odblokowanie wykonania

- [x] Uruchomić zadanie w sesji z dostępem do GPU. Obecny sandbox zwraca błąd
  inicjalizacji NVML, a próba eskalacji kończy się błędem usługi autoryzacyjnej.
  **2026-09-20:** `nvidia-smi` i `torch.cuda` działają; rekonstruktor T5 idzie na CPU
  (`LLEAKM_RECONSTRUCT_DEVICE=cpu`, `--reconstruct-device cpu`), serwer ofiary na `:8010`.
- [x] Zapewnić lokalnie wszystkie checkpointy z
  `experiment_validation/models_publication.json`.
  Pobrane: Qwen2.5-1.5B, Qwen2.5-3B, TinyLlama, phi-1.5, Phi-3.5-mini,
  Llama-3.2-3B-Instruct, gemma-2-2b-it, T5 First/Middle.
- [x] Sprawdzić budżet VRAM. Serwer ofiary i dwa modele T5 są obecnie ładowane
  równocześnie; przy GPU 8 GB może być konieczne uruchomienie rekonstruktora na
  CPU albo rozdzielenie zbierania śladów i rekonstrukcji na dwa etapy.
- [x] Wykonać smoke test jednego modelu i 2–3 promptów. Potwierdzić, że katalog
  wyniku zawiera aktualizowane na bieżąco `samples.jsonl`, `progress.json` i
  `config.json`.
  Katalog: `experiment_validation/results/run_smoke_20260920_tinyllama_1_1b_full`
  (`status=complete`, 3/3, `finish_reason` length/eos, `complete_only` w `summary.json`).
  `progress.json` jest od teraz zapisywany po każdym prompcie.

## P1 — macierz długich odpowiedzi

- [x] Uruchomić siedem modeli, 300 wspólnych promptów, limit 512 tokenów,
  generowanie deterministyczne i rekonstrukcję wszystkich segmentów.
  Katalogi `run_full_20260920_*`, wszystkie `status=complete`, 300/300.

Lokalna piątka complete. Llama/Gemma wznowione 2026-09-22 (`models_publication.json --resume`):

```bash
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_model_matrix.py \
  --models-file experiment_validation/models_local.json \
  --campaign-id full_20260920 \
  --prompts 300 \
  --max-new-tokens 512 \
  --samples-per-segment 3 \
  --max-sentences 0 \
  --num-first-candidates 3 \
  --prompt-format legacy \
  --reconstruction-seed 20260706 \
  --hf-offline \
  --reconstruct-device cpu \
  --port 8010 \
  --resume
```

Pełna siódemka po `huggingface-cli login`:

```bash
HF_HUB_OFFLINE=1 python experiment_validation/scripts/run_model_matrix.py \
  --models-file experiment_validation/models_publication.json \
  --campaign-id full_20260920 \
  --prompts 300 \
  --max-new-tokens 512 \
  --samples-per-segment 3 \
  --max-sentences 0 \
  --num-first-candidates 3 \
  --prompt-format legacy \
  --reconstruction-seed 20260706 \
  --hf-offline \
  --reconstruct-device cpu \
  --port 8010 \
  --resume
```

- [ ] Po przerwaniu wznowić dokładnie tę samą kampanię przez dodanie `--resume`.
- [ ] Nie zmieniać `campaign-id` przy wznowieniu.
- [ ] Monitorować wolne miejsce, pamięć GPU/RAM i katalog `logs/`; nie usuwać
  logów, ponieważ są potrzebne do późniejszych powtórzeń i kalibracji ataku.

## P2 — kontrola kompletności odpowiedzi

- [x] Dla każdego z siedmiu katalogów potwierdzić:
  `config.status == "complete"`, 300 unikalnych indeksów oraz
  `progress.done == progress.total == 300`.
- [x] Potwierdzić w konfiguracji: `max_new_tokens=512`, `max_sentences=0`,
  `temperature=0`, `top_p=1` i właściwy `prompt_format`.
- [x] Policzyć i opublikować odsetek `finish_reason=eos` osobno dla każdego
  modelu. EOS% per model w Table `tab:eos_completeness`.
  (Rozbicie EOS×temat nie jest osobną tabelą; topic means dotyczą φ.)
- [x] Do tabeli „complete answers” używać wyłącznie bloku
  `summary.json -> complete_only`; rekordów zakończonych przez limit długości
  nie wolno nazywać pełnymi odpowiedziami.
  Analiza używa `response_complete` z `samples.jsonl` (zgodne z `complete_only`).
- [x] Raportować również liczbę odpowiedzi odrzuconych przez filtr EOS oraz
  analizę wrażliwości „wszystkie odpowiedzi vs EOS-only”, aby ujawnić możliwy
  błąd selekcji.
- [x] Jeżeli w którymś temacie są mniej niż dwie odpowiedzi EOS-complete,
  zwiększyć limit i powtórzyć cały model zamiast dobierać wygodne przypadki.
  Phi-3.5-mini: $n_{\mathrm{EOS}}=0$ przy 512; brak complete-only claim; nie
  dobierano wygodnych przypadków.

## P3 — powtórzenia rekonstruktora i wspólne seedy

- [x] Na nowych, kompletnych logach wykonać pięć seedów:
  `20260706, 20260707, 20260708, 20260709, 20260710`.
- [x] Dla porównań między warunkami używać tego samego seedu dla tego samego
  promptu. `robustness_eval.py` realizuje to przez stabilny BLAKE2s, niezależny
  od `PYTHONHASHSEED`.
- [x] Najpierw uruchomić plan i zweryfikować liczbę wywołań (3300).
- [x] Uruchomić właściwy przebieg bez `--plan`. 22 kompletne komórki, 3300
  rekonstrukcji. `--complete-only` odrzucone: kalibracja EOS-only 137/167 < 200.
  Użyto 30 test / 270 kalibracji, `max_sentences=1`, jeden stochastyczny sample.
- [x] `analysis/robustness_eval.json -> complete` ma wartość `true`.
- [x] Raportować średnią, crossed-bootstrap 95% CI oraz zmienność średnich
  między seedami; nie traktować powtórzeń seedów jako niezależnych promptów.

## P4 — adaptacyjny atakujący

- [x] Zachować rozłączność zbiorów: 30 promptów testowych i 270 śladów
  kalibracyjnych na model.
- [x] Ocenić osobno `frozen` i `adaptive` dla `bucket_8`, `pad_32`, `batch_2`,
  `batch_4` i `rand_pad_8`; baseline ma jeden wspólny wariant.
- [x] Opisać atak precyzyjnie jako „prompt-disjoint empirical MAP inverse plus
  unchanged T5 decoder”. Nie nazywać go pełnym ponownym trenowaniem T5.
- [x] Sprawdzić, czy adaptacja odzyskuje część wyniku względem zamrożonego
  atakującego, oraz raportować również przypadki, w których jej nie odzyskuje.
  Odzysk: bucket/batch. Brak odzysku inflacji `pad_32`.
- [ ] Jeżeli recenzent wymaga retreningu end-to-end, dodać osobny eksperyment
  fine-tuningu T5 na przekształconych śladach; obecna kalibracja MAP go nie
  zastępuje.

## P5 — aktualizacja manuskryptu po uzyskaniu wyników

- [x] Nie zmieniać abstraktu ani wniosków na podstawie smoke testu lub wyniku
  częściowego. (Aktualizacja po kompletnej siódemce `full_20260920`.)
- [x] Uzupełnić Methods o: kryterium EOS, pięć seedów, common random numbers,
  rozłączny split kalibracja/test i dokładną wiedzę adaptacyjnego atakującego.
- [x] Dodać tabelę pełnych odpowiedzi oraz tabelę `frozen` vs `adaptive` z
  liczebnościami, CI i narzutem obron.
  Tabele `tab:eos_completeness` i `tab:robustness`. Narzut obron zostaje w
  pilocie 96-tokenowym.
- [x] Zaktualizować Abstract, Results, Discussion, Limitations, Conclusion oraz
  `article1/COVER_LETTER.md` wyłącznie liczbami z kompletnych artefaktów.
- [x] Nie usuwać nadal aktualnych ograniczeń: oracle długości aplikacyjnych nie
  jest pasywnym atakiem na TLS/HTTP2/QUIC, a miary leksykalne nie dowodzą
  zgodności semantycznej.
- [ ] Usunąć z `article1/SUBMISSION_CHECKLIST.md` pozycje dotyczące wykonania
  dopiero po zakończeniu i weryfikacji kampanii.

## P6 — reprodukowalność i kontrola końcowa

- [x] Uruchomić testy i kontrolę składni protokołu (część P6, przed kampanią):

```bash
python -m unittest discover -s experiment_validation/tests -v
python -m py_compile app.py weiss_reconstruction.py experiment_validation/scripts/*.py
git diff --check
```

- [x] Uruchomić wszystkie analizy i generatory tabel/wykresów od zera.
- [ ] Dodać nowe artefakty do audytu treści oraz manifestu SHA-256.
- [ ] Uruchomić:

```bash
python experiment_validation/scripts/audit_release_content.py
python experiment_validation/scripts/verify_manuscript_numbers.py
python experiment_validation/scripts/make_repro_manifest.py
```

- [ ] Skompilować `article1/template.tex`, sprawdzić ostrzeżenia LaTeX oraz
  przejrzeć finalny PDF strona po stronie.
- [ ] Wykonać niezależną kontrolę wszystkich liczb w abstrakcie, tabelach i
  wnioskach względem JSON/JSONL.
- [ ] Oznaczyć release, zarchiwizować go (np. Zenodo) i wpisać DOI do Data
  Availability Statement.

## Nadal wartościowe, lecz odrębne rozszerzenia

- [ ] Dodać co najmniej jeden autoryzowany eksperyment z przechwyceniem
  zaszyfrowanego ruchu, zamiast zakładać bezpośrednią obserwację ramek SSE.
- [ ] Dodać ocenę ludzką, odzysk encji lub jedną ustaloną miarę
  paraphrase-aware obok miar leksykalnych.

## Definicja zakończenia

Praca jest gotowa do deklarowania nowych wyników dopiero wtedy, gdy pełna
macierz ma kompletne artefakty, analiza pełnych odpowiedzi obejmuje wyłącznie
EOS-complete, eksperyment wieloseedowy ma `complete: true`, nie ma przecieku
między kalibracją a testem, liczby w manuskrypcie przechodzą automatyczną
weryfikację, a finalny PDF został sprawdzony przez obu autorów.
