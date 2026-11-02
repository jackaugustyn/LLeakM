# Nowe referencje TinyLlama i Phi-3.5

Przygotowano osobny eksperyment generacji dla 300 archiwalnych promptów każdego modelu. Nie odtworzono historycznych token IDs ani nie potwierdzono tożsamości historycznych rewizji modeli. Nowe generacje używają jawnie przypiętych, obecnie dostępnych checkpointów; zachowują archiwalny format `legacy`, dekodowanie zachłanne, `temperature=0`, `top_p=1` i limit 512 tokenów.

Runner `generate_fresh.py` ładuje model ofiary bez T5 i wywołuje naprawiony handler z `app.py` wewnątrz procesu. Sprawdza każdy wyemitowany fragment, dokładne token IDs, pełne dekodowanie referencji oraz sumy bajtów i hashe. To instrumentowany eksperyment długości payloadów, bez pomiaru rzeczywistego transportu HTTP, pakietów czy opóźnień sieciowych. Fragmenty są przygotowane po wygenerowaniu całej sekwencji, zgodnie z aktualnym harness.

## Uruchomienie po zwolnieniu GPU przez T5

Z katalogu głównego repozytorium:

```bash
python3 experiment_validation/revalidation_20261031/fresh_references/check_cache.py --download

python3 experiment_validation/revalidation_20261031/fresh_references/generate_fresh.py --label tinyllama_1_1b --output experiment_validation/revalidation_20261031/fresh_references/tinyllama

python3 experiment_validation/revalidation_20261031/fresh_references/generate_fresh.py --label phi_3_5_mini --output experiment_validation/revalidation_20261031/fresh_references/phi --placement auto_offload --gpu-memory-gib 6 --cpu-memory-gib 5
```

Nie uruchamiać generacji ofiar równocześnie z T5 na tej karcie. Każde polecenie uruchamia jeden model w osobnym procesie. `--limit 1` pozwala wykonać próbę jednego promptu, a kolejne wywołanie bez limitu wznawia zapisane wyniki. Niepełny lub uszkodzony rekord, zmiana kodu, parametrów, checkpointu, wejścia czy faktycznej mapy urządzeń blokuje mieszanie wyników.

`check_cache.py` domyślnie sprawdza wyłącznie pliki lokalne. Opcja `--download` pobiera brakujące pliki z przypiętych rewizji; nie pobiera kodu zdalnego ani alternatywnych kopii wag. Raport sprawdza rozmiary, SHA256 oraz udostępnione przez repozytoria hashe LFS.

## Pamięć i dokładność

- TinyLlama: 2,20 GB wag; całość na CUDA w FP16.
- Phi-3.5: 7,64 GB wag FP16, zbyt mały zapas pamięci przy pełnym umieszczeniu na 8 GB GPU. Plan `auto_offload` ogranicza budżet wag GPU do 6 GiB i dopuszcza 5 GiB RAM. Analiza modelu na tensorach meta umieszcza 26 bloków na GPU, a 6 bloków i głowicę w pamięci CPU; rzeczywista mapa zapisywana jest po załadowaniu. Accelerate przenosi potrzebne wagi podczas wykonania. Kwantyzacja nie jest używana.
- Zgodność modelu z 8 GB podczas pełnej generacji pozostaje do sprawdzenia po zwolnieniu GPU. Bufor KV, aktywacje i transfery offloadu zwiększają wymagania i czas. Runner zapisuje błąd i nie zmienia automatycznie precyzji lub protokołu.
- Zmiana rewizji, środowiska wykonania i mapy urządzeń ogranicza interpretację różnic względem archiwum. Nowych danych nie należy przedstawiać jako czystego efektu naprawy detokenizacji.

## Wyniki i rekonstrukcja

Każdy katalog modelu zawiera `protocol.json`, `runtime.json`, `status.json`, `summary.json`, nowe logi w `logs/` i `canonical_traces.jsonl`. Kompletność wymaga dokładnie 300 rekordów. Dopiero wtedy można uruchomić T5 na nowym pliku:

```bash
python3 experiment_validation/revalidation_20261031/inference/rerun_canonical.py --input experiment_validation/revalidation_20261031/fresh_references/tinyllama/canonical_traces.jsonl --output experiment_validation/revalidation_20261031/inference/fresh_tinyllama_cuda --device cuda

python3 experiment_validation/revalidation_20261031/inference/rerun_canonical.py --input experiment_validation/revalidation_20261031/fresh_references/phi/canonical_traces.jsonl --output experiment_validation/revalidation_20261031/inference/fresh_phi_cuda --device cuda
```

Pola `source_kind=fresh_contextual_reference`, `historical_comparison_available=false` i powiązania z archiwalnymi promptami oznaczają nowy eksperyment. `old_pred_full_text` jest pustym polem zgodności ze wspólnym runnerem: wyników `old_phi` nie interpretować jako historycznego pomiaru ani punktu odniesienia. Wspólny runner T5 ma konserwatywny, archiwalny opis zakresu w protokole; bezpośrednia weryfikacja nowych token IDs i dekodowania znajduje się w logach i osobnym walidatorze wyników.

## Wykonane sprawdzenia przed generacją

- Dwa plany CLI potwierdzają po 300 dokładnych indeksów i zgodność archiwalnych ustawień.
- Cztery testy przechodzą: identyfikatory i długości, odrzucanie uszkodzonych ramek/referencji, zgodność przy wznowieniu, pełna ścieżka CLI z prawdziwym handlerem i zamockowanymi wagami/CUDA. Test CLI obejmuje jawne ustawienie offloadu i wznowienie bez powtórzenia pierwszej próbki.
- Rzeczywiste przypięte tokenizery obu modeli poprawnie przeszły 8 przypadków tekstowych, w tym spacje, CR/LF i wielobajtowy znak Unicode. Nie jest to test generacji wag modelu.
- `phi_memory_plan.json` pochodzi z tensorów meta: nie załadowano wag ofiary na GPU i nie wykonano generacji w ramach przygotowania.

Ostateczny stan pobrania wag znajduje się w `cache_validation.json`. Stan w chwili przekazania przygotowania zapisano w `readiness.json`.
