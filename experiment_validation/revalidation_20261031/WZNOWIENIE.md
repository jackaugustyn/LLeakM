> **Aktualizacja:** kolejka obliczeń została ukończona 1 listopada 2026. Bieżący stan: [AKTUALNY_STAN.md](AKTUALNY_STAN.md). Poniższy zapis dotyczy wcześniejszej pauzy.

# Praca wstrzymana na prośbę użytkownika — 31 października 2026

Użytkownik: „zakończ chwilowo robotę, potem uruchomię cię tak byś mógł uruchamiać sesje”. Nie uruchamiać dalszych obliczeń do wznowienia przez użytkownika.

## Zadanie do kontynuacji

Sprawdzić wszystkie wyniki artykułu i wykonać konieczne ponowne obliczenia, aby uzyskać wiarygodne dane. Użytkownik upoważnił do obliczeń i niezbędnych napraw kodu. Docelowo uwzględnić zweryfikowane zmiany w artykule, nie przedstawiając częściowych rekonstrukcji jako pełnej kampanii.

## Zakończone

- Przeliczono 2100 głównych par tekstów i ich pierwsze zdania, 240 kontroli, 420 rekordów obron i 3300 MAP. Sprawdzone metryki tekstowe zgadzają się dokładnie; niezależnie potwierdzono 17/21 testów modeli i 10/10 testów obron. Raport: `statistics/WALIDACJA_STATYSTYK.md`.
- Analiza 60 scenariuszy zamiast 300 niezależnych wariantów zachowuje te same 17 istotnych par; nowe przedziały zapisano w `statistics/scenario_cluster_sensitivity.json`.
- Narzut historycznego `rand_pad_8` nie odtwarza się z receptury ziaren. Obliczono analityczną **wartość oczekiwaną**: Qwen 79,755339%, Llama 83,133614%. Nie jest to odzyskana realizacja historyczna.
- Naprawiono percentyle w `trace_baselines.py`, ogony p i przypadek równych sum rang w `analyze_stats.py` i `defense_eval.py`. Oryginalnych analiz nie nadpisano.
- Kolektor `run_validation.py` używa jawnego, zweryfikowanego oracle'a długości z logu; zapisuje dokładne wektory i hashe, sprawdza sumę odebranych bajtów i odrzuca mieszanie starych rekordów. Domyślna metryka CLI to kosinus TF. Poprawiono powiązany finalizer, a stary `client.py` oznaczono ostrzeżeniem.
- Odtworzono 2100 kanonicznych wektorów długości w `collection/canonical_traces.jsonl`. Dotyczą zapisanych emitowanych tekstów, nie naprawionych referencji. Potwierdzono 2073 różnice śladów i 764 różnice liczby segmentów względem replay starego kolektora.
- Naprawiono kontekstową detokenizację w `app.py`, buforowanie UTF-8, zgodność fragmentów z pełnym dekodowaniem oraz zapis token IDs i pochodzenia nowych logów. W starych 2100 logach nie ma token IDs. TinyLlama/Phi-3.5 nadal wymagają nowych generacji lub oryginalnych IDs; brakujące spacje nie zostały dodane heurystycznie. Raport: `references/REFERENCE_VALIDATION.md`.
- Naprawiono `weiss_reconstruction.reconstruct`: ziarnowanie odbywa się po ładowaniu modeli, aby stan zimnego/ciepłego cache nie zmieniał próby. Osobny test wznowienia przeszedł.
- Pełny zestaw 22 istniejących i nowych testów przeszedł przed ostatnią poprawką ziarnowania; po niej przeszedł dodatkowy test `test_reconstruction_seed_resume`.
- Runner `inference/rerun_canonical.py` przeszedł przegląd oraz pełną próbę CPU jednej odpowiedzi: 1/2100, 35/35 segmentów, φ=0,516410 wobec starego 0,538608. Wynik w `inference/cpu_verified_smoke/` jest **częściowy** i nie zastępuje tabeli. Wcześniejsze `cpu_probe` i `cpu_full_smoke` są próbami rozwojowymi, nie wynikami kampanii.

## GPU i pierwsze czynności po wznowieniu

Użytkownik potwierdził, że w zwykłym terminalu WSL działa NVIDIA GeForce RTX 3070 z 8192 MiB VRAM (207 MiB zajęte). W poprzedniej sesji brakowało `/dev/dxg`; PyTorch widział zero CUDA, a `nvidia-smi` zgłaszało blokadę systemową. Eskalacja kontroli GPU nie została wykonana z powodu błędu 404 infrastruktury automatycznej autoryzacji, nie z powodu uznania działania za niebezpieczne. Nie obchodzić tej kontroli.

Po wznowieniu sprawdzić dostęp CUDA i istniejący `inference/cuda_full/status.json` (użytkownik nie potwierdził uruchomienia przed pauzą). Gdy GPU jest dostępne:

```bash
cd /home/paugustynowicz/LLeakM
python3 experiment_validation/revalidation_20261031/inference/rerun_canonical.py --output experiment_validation/revalidation_20261031/inference/cuda_full --device cuda
```

Runner używa lokalnych, przypiętych checkpointów T5, float32, wszystkich segmentów i trzech kandydatów; zapisuje rzeczywiste wejścia enkodera i hashe wag. Wznowienie wymaga tego samego kodu, protokołu, wejść i urządzenia. Nie zmieniać jego pliku po rozpoczęciu kampanii bez nowego katalogu/protokołu. Uszkodzona końcowa linia JSONL zatrzymuje wznowienie; nie jest automatycznie usuwana. CPU i CUDA mają odrębne wyniki.

## Nadal otwarte

1. Dokończyć 2100 rekonstrukcji z kanonicznych długości i przeliczyć ich wszystkie metryki/statystyki. Nie przypisywać całej zmiany względem archiwum wyłącznie kolektorowi: środowisko i stan losowania także mogą się różnić.
2. Wykonać nowe generacje co najmniej TinyLlama i Phi-3.5 z naprawioną detokenizacją. Ich wagi i tokenizery nie były dostępne w widocznym cache; dostępne były Gemma i oba T5. Najpierw ustalić zasoby i przypiąć wersje. Nie utożsamiać nowych fragmentów kontekstowych ze starym protokołem niezależnego dekodowania tokenów.
3. Dalsze przeliczenia kontroli i MAP powinny zachować jawny zakres: 96 tokenów/do trzech segmentów versus 512 tokenów/pierwszy segment względem całej referencji. Ablacja ze stałym pokryciem wejścia jest osobnym eksperymentem, nie prostym przeliczeniem obecnej tabeli.
4. Agent `recompute_statistics` został przerwany podczas aktualizacji artykułu i README. Sprawdzić `article_updates/`, bieżący diff i `pause_state.json`; możliwe są częściowe zmiany tekstu. PDF nie był ponownie kompilowany w tym etapie. Dokończyć opis analizy scenariuszowej, oznaczenie oczekiwanego narzutu paddingu oraz rozróżnienie naprawionego kodu i starych wyników, następnie skompilować i sprawdzić PDF.
5. Sprawdzić końcową spójność kodu/testów, wyników i artykułu. Stary snapshot `analysis/revision_snapshot_20261031.json` opisuje poprzednią rewizję tekstu; nie nadpisywać go jako dowodu obecnego stanu.

Zmiany użytkownika sprzed tej pracy obejmowały już artykuł, PDF i dokumenty po poprzedniej rewizji. Nie resetować drzewa roboczego. Hashy stanu w chwili pauzy szukać w `pause_state.json`. Nie wykonano zewnętrznej publikacji, nowych generacji ofiar ani pełnej kampanii CUDA.
