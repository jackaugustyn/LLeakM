# Niezależna weryfikacja rekonstrukcji T5

Skrypt nie ładuje wag do RAM ani GPU. Odczytuje je strumieniowo do sprawdzenia SHA-256, a zapisane wejścia odtwarza bezpośrednio lokalnymi tokenizerami.

Pełna kampania główna:

```bash
python3 experiment_validation/revalidation_20261031/inference_validation/verify_inference.py \
  --run experiment_validation/revalidation_20261031/inference/cuda_full \
  --output experiment_validation/revalidation_20261031/inference_validation/cuda_full
```

Inna pełna kampania korzystająca z kanonicznego inwentarza, w tym nowe 300 lub 600 referencji:

```bash
python3 experiment_validation/revalidation_20261031/inference_validation/verify_inference.py \
  --run SCIEZKA_DO_REKONSTRUKCJI \
  --input SCIEZKA_DO_KANONICZNYCH_REFERENCJI.jsonl \
  --output OSOBNY_KATALOG_WERYFIKACJI
```

Każda obecna seria musi mieć 300 unikalnych indeksów i 60 scenariuszy po 5 wariantów. Domyślny inwentarz główny dodatkowo wymaga wszystkich siedmiu modeli. Brak choćby jednego rekordu, duplikat, zmieniony protokół, niewłaściwe hashe, błędne wejście enkodera lub niespójna metryka nie pozwalają uzyskać statusu `VERIFIED_COMPLETE`.

`--allow-partial` umożliwia diagnostyczny odczyt rosnącego pliku. Wynik pozostaje `PARTIAL_DIAGNOSTIC`, a pełne statystyki nie są generowane. Bez tej flagi częściowy przebieg kończy się kodem 2. Błąd integralności kończy się niezerowym kodem i raportem `FAILED`. Katalog wynikowy musi być inny od źródłowego katalogu inferencji; archiwów nie wolno nadpisywać.

Weryfikacja obejmuje:

- powiązanie logów generacji, referencji, wektorów długości, promptów, indeksów, ziaren i źródeł;
- zgodność segmentacji, kontekstu, serializacji długości, wszystkich zapisanych token IDs enkodera i obcinania do 80 tokenów;
- niezależne metryki TF, ED, ROUGE-1 i ROUGE-L dla całego tekstu i pierwszego zdania;
- jawne liczenie wcześniejszych przerwań po pustym kandydacie i nieprzetworzonych segmentów;
- w nowych referencjach: dokładne ID wygenerowanych tokenów, dekodowanie pełnej sekwencji przypiętym tokenizerem, identyczność emitowanych ramek i SHA-256 checkpointów ofiary;
- po pełnym ukończeniu: statystyki modeli, EOS, LR@0,5, tematów, bootstrap promptów, bootstrap scenariuszy i testy sparowane z korekcją Holma.

W nowych generacjach puste `old_pred_full_text` jest wyłącznie polem zgodności z runnerem; `old_metrics` i `delta_phi` w analizie mają wartość `null`. Nie traktuje się zera z tego pola jako historycznej rekonstrukcji. W głównej kampanii stare referencje TinyLlama/Phi-3.5 pozostają diagnostyczne.

Testy:

```bash
python3 -m unittest discover -s experiment_validation/tests -v
python3 -m unittest discover -s experiment_validation/revalidation_20261031/inference_validation -p 'test_*.py' -v
```

Zestaw testów weryfikatora korzysta z prawdziwej wcześniejszej próby CPU i celowo psuje referencję, metrykę, liczbę segmentów, token enkodera oraz kontekst. Osobny test dowodzi odrzucenia referencji zgodnej z tekstami kroków, lecz niezgodnej z zapisanymi ID tokenów.
