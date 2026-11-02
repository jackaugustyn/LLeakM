# Walidacja detokenizacji i referencji — 31 października 2026

Sprawdzono wszystkie 2100 rekordów głównej kampanii oraz odpowiadające im logi. Nie wykonano generacji modelu ani nie zastąpiono zapisanych referencji tekstem odtworzonym heurystycznie.

## Wynik przeliczenia

| Model | Logi | Referencje bez spacji ASCII | Logi z identyfikatorami wygenerowanych tokenów |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 300 | 0 | 0 |
| Qwen2.5-3B | 300 | 0 | 0 |
| Llama-3.2-3B | 300 | 0 | 0 |
| Gemma-2-2B | 300 | 0 | 0 |
| Phi-1.5 | 300 | 0 | 0 |
| TinyLlama-1.1B | 300 | 299 | 0 |
| Phi-3.5-mini | 300 | 162 | 0 |

W każdym rekordzie zapisany tekst odpowiedzi zgadza się z konkatenacją fragmentów w logu. Każda zapisana długość fragmentu zgadza się z liczbą jego bajtów UTF-8. Żadna z 2100 referencji nie zawiera znaku zastępczego U+FFFD. Są to kontrole spójności zapisu; nie dowodzą poprawnej detokenizacji całej wygenerowanej sekwencji.

W skonfigurowanym lokalnym cache Hugging Face dostępny był tokenizer ofiary Gemma oraz dwa tokenizery T5. Tokenizery TinyLlama i Phi-3.5 nie były tam dostępne. W dostarczonych rekordach głównej kampanii nie ma ani list oryginalnych token IDs, ani identyfikatorów tokenów w poszczególnych krokach. Nie ma zatem podstaw do odtworzenia brakujących odstępów w tych dwóch seriach jako pewnej referencji.

## Niejednoznaczność odwrotnego mapowania

Dla lokalnego tokenizera Gemma sprawdzono cały słownik 256000 identyfikatorów. W każdej z 300 sekwencji występuje co najmniej jeden fragment mający więcej niż jeden odpowiadający mu token ID; łącznie dotyczy to 29045 kroków. Przykładowo zwykła spacja ma dwa kandydaty: identyfikator tokenu znakowego i identyfikator bajtu ASCII.

Niejednoznaczność ID nie oznacza automatycznie wadliwej referencji. Dwie sprawdzone wersje odwrotnego mapowania — wybór pierwszego i ostatniego kandydata dla każdego fragmentu — dały różne listy IDs dla wszystkich 300 odpowiedzi, ale pełne dekodowanie obu wersji zgadzało się z każdą zapisaną referencją Gemma. Nie jest to sprawdzenie wszystkich kombinacji ani poświadczenie historycznej wersji tokenizera. Żadnej z tych odtworzonych list nie zapisano jako danych pierwotnych.

## Naprawa aparatury

W `app.py` zastąpiono niezależne dekodowanie pojedynczych tokenów dekodowaniem narastających prefiksów. W ten sposób token oznaczający początek słowa zachowuje odstęp w kontekście, a nie jest traktowany jako samodzielny początek odpowiedzi. Niekompletne bajty UTF-8 są buforowane; dany krok może emitować pusty fragment. Ostatni krok zachowuje także rzeczywisty końcowy znak zastępczy, jeśli pełne dekodowanie go zawiera.

Przed rozpoczęciem wysyłania odpowiedzi program sprawdza, że fragmenty dokładnie składają się na pełne dekodowanie tej samej sekwencji IDs. Niezgodny, niestabilny prefiksowo decoder powoduje błąd zamiast emisji zmienionego tekstu. Zachowano jeden zapis kroku na każdy wygenerowany token poza końcowym EOS. Długości są teraz długościami fragmentów przypisanych do kroków, włącznie z zerami, a nie długościami samodzielnie dekodowanych tokenów. To zmiana protokołu, której nie wolno przypisać historycznym wynikom T5.

Nowe logi zawierają pełną listę IDs, ID każdego kroku, nazwę polityki dekodowania, klasę i ślad SHA-256 backendu tokenizera, dostępne identyfikatory rewizji oraz SHA-256 źródła aparatury odczytany przy imporcie. Brakujące rewizje pozostają `null`. Log zapisuje się atomowo przed komunikatem `[DONE]`.

Serwer już wcześniej generował pełną listę tokenów przed emisją strumienia. Obecna walidacja korzysta z tego modelu wykonania; nie przedstawia aparatury jako przetestowanego produkcyjnego dekodera działającego w czasie rzeczywistym.

## Sprawdzenia poprawki

Dziewięć testów zakończyło się powodzeniem: sześć testów detokenizacji i zapisu odpowiedzi oraz trzy istniejące testy doboru pamięci podręcznej Gemma. Testy używają rzeczywistego dekodera biblioteki `tokenizers` do odtworzenia problemu odstępów oraz bajtów UTF-8, bez ładowania wag modelu. Sprawdzono również cztery przykłady przez lokalny `GemmaTokenizerFast`; wszystkie nowe fragmenty zgadzają się z pełnym dekodowaniem. Dla tekstu `A 🫨 B` stare dekodowanie pojedynczych tokenów daje inny wynik, podczas gdy nowa metoda zachowuje pełny tekst, buforując trzy kroki.

Polecenia odtwarzające:

```bash
python3 -m unittest experiment_validation.tests.test_stream_detokenization experiment_validation.tests.test_app_hybrid_cache -v
python3 experiment_validation/revalidation_20261031/references/audit_references.py
```

Wyniki szczegółowe i hashe wejść: `reference_audit.json`. Poprzedni kod zachowano w `app_before.py`. Historia danych nie została zmieniona. Naprawa referencji TinyLlama/Phi-3.5 i pełne porównanie zwykłych odpowiedzi nadal wymagają oryginalnych IDs z właściwym tokenizerem albo nowych generacji.
