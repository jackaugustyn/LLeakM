## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-01T18:29:12.682031+00:00
- Verification Status: ANALYZED
- Version Label: eos_selector_inference_validation_v1

## Weryfikacja nowych rekonstrukcji T5

Stan: **PARTIAL_DIAGNOSTIC**, zapisane rekordy **2/2100**.

Sprawdzono hashe źródeł i checkpointów, powiązanie wszystkich wejść z logami, unikalność rekordów, segmentację, rzeczywiste serializacje i identyfikatory enkodera oraz metryki obliczone niezależnie z tekstów.

Wywołania enkodera: 53; obcięte do 80 tokenów: 5; odpowiedzi z nieprzetworzonymi końcowymi segmentami: 0.

Historyczne referencje TinyLlama i Phi-3.5 mają status diagnostyczny; nowych generacji nie utożsamia się z naprawionymi historycznymi odpowiedziami. Zmianę selektora bada osobny audyt kandydatów i sparowane porównanie z poprzednią kampanią.

Wynik jest częściowy; nie wygenerowano agregatów pełnej kampanii.
