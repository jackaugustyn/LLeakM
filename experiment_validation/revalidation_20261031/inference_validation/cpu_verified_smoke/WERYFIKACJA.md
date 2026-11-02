# Weryfikacja nowych rekonstrukcji T5

Stan: **PARTIAL_DIAGNOSTIC**, zapisane rekordy **1/2100**.

Sprawdzono hashe źródeł i checkpointów, powiązanie wszystkich wejść z logami, unikalność rekordów, segmentację, rzeczywiste serializacje i identyfikatory enkodera oraz metryki obliczone niezależnie z tekstów.

Wywołania enkodera: 35; obcięte do 80 tokenów: 0; odpowiedzi z nieprzetworzonymi końcowymi segmentami: 0.

Historyczne referencje TinyLlama i Phi-3.5 mają status diagnostyczny; nowych generacji nie utożsamia się z naprawionymi historycznymi odpowiedziami. Nie przypisuje się całej zmiany wyniku wyłącznie naprawie kolektora.

Wynik jest częściowy; nie wygenerowano agregatów pełnej kampanii.
