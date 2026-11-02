# Wyniki wznowionych obliczeń

## Material Passport

- Origin Skill: ARS-Codex / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-01T15:15:40.188789+00:00
- Verification Status: VERIFIED — obliczenia w ramach jawnie zapisanych protokołów.
- Version Label: revalidation_20261031_completed

Ukończono i sprawdzono 2100 rekonstrukcji z archiwalnych, zweryfikowanych długości, 3900 rekonstrukcji kontroli/obron/MAP oraz 600 nowych generacji ofiar i ich 600 rekonstrukcji T5. Nowe generacje mają własne wersje checkpointów i zweryfikowane dekodowanie całej sekwencji; nie są odzyskanymi historycznymi odpowiedziami.

| Seria | Model | N | Średnia φ |
|---|---|---:|---:|
| cuda_full | qwen_1_5b | 300 | 0.535463 |
| cuda_full | qwen_3b | 300 | 0.525143 |
| cuda_full | llama_3_2_3b | 300 | 0.514621 |
| cuda_full | gemma_2_2b | 300 | 0.482001 |
| cuda_full | phi_1_5 | 300 | 0.516368 |
| cuda_full | tinyllama_1_1b | 300 | 0.079159 |
| cuda_full | phi_3_5_mini | 300 | 0.121017 |
| fresh_tinyllama_cuda | tinyllama_1_1b | 300 | 0.435205 |
| fresh_phi_cuda | phi_3_5_mini | 300 | 0.504151 |

Pełne metryki, przedziały i testy par znajdują się w odpowiednich katalogach `inference_validation/*/statistics.json`. Wyniki pilotaży i MAP: `secondary/cuda_full/summary.json`.

Ważne granice interpretacji: dwie archiwalne serie nadal mają wadliwe referencje; nowe referencje są osobnym eksperymentem. Zmiana kolektora, detokenizacji, środowiska i wersji wag nie została rozdzielona przyczynowo. Limit 80 tokenów enkodera nadal ogranicza obserwację. Metryki leksykalne nie mierzą bezpośrednio ujawnienia danych.

Artykuł wymaga teraz uzgodnienia z kompletnymi wynikami. Kolejka nie zmienia automatycznie wniosków ani nie łączy nowych odpowiedzi z historycznym rankingiem.
