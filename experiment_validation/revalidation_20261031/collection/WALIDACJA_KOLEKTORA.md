# Walidacja obserwacji długości — 31 października 2026

Odtworzono **2100 wektorów długości** z siedmiu głównych przebiegów, po 300 rekordów na model. Dla każdego zweryfikowano zgodność promptu i identyfikatora uruchomienia, modelu, kolejności kroków, długości UTF-8 fragmentu, długości zapisanego strumienia, liczby wygenerowanych tokenów, zakończenia EOS oraz zgodności referencji ze złączeniem fragmentów. Wszystkie te kontrole przeszły.

Plik [canonical_traces.jsonl](canonical_traces.jsonl) zawiera dokładne wektory `token_lengths`, zachowane referencje `reference`, poprzednie rekonstrukcje `old_pred_full_text`, oryginalne ziarna rekonstruktora, identyfikatory rekordów i hashe danych źródłowych. Może być wejściem do nowych rekonstrukcji. **Nie zawiera nowych wyników T5 ani odtworzonych, poprawnie detokenizowanych referencji.**

## Porównanie ze starym kolektorem

Replay dotychczasowej logiki `iter_lines()` odrzuca bajty CR/LF wewnątrz tokenów. W porównaniu z długościami zapisanymi przez serwer zmienia 2073 z 2100 śladów i liczbę segmentów w 764 próbkach. Zgodność liczby jednostek nie wykrywa tego błędu.

| Model | Zmienione ślady | Zmieniona liczba segmentów | Referencje bez spacji ASCII |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 282 | 149 | 0 |
| Qwen2.5-3B | 300 | 227 | 0 |
| Llama-3.2-3B | 291 | 204 | 0 |
| Gemma-2-2B | 300 | 23 | 0 |
| Phi-1.5 | 300 | 14 | 0 |
| TinyLlama | 300 | 57 | 299 |
| Phi-3.5-mini | 300 | 90 | 162 |

Jest to porównanie z replay obecnie znanej starej logiki, a nie dowód odzyskania historycznych wejść T5. Obecność poprawnych liczników bajtów nie poświadcza zgodności tekstu z dekodowaniem pełnej sekwencji tokenów. Dwie serie z wadliwymi granicami słów pozostają oznaczone `diagnostic_series: true`. We wszystkich siedmiu seriach referencje nie zawierają znaku zastępczego Unicode U+FFFD; nie dowodzi to poprawnej detokenizacji.

W archiwum występuje również 256 pustych fragmentów TinyLlama i 210 Phi-3.5-mini. Odtworzone wektory zachowują odpowiadające im zera; usunięcie ich zmieniłoby obserwację.

## Naprawa bieżącej ścieżki

`run_validation.py` korzysta teraz jawnie z instrumentowanego źródła długości `validated-log-payload-byte-lengths-v1`. Odbiera cały strumień przez `iter_content()`, bez wyznaczania granic z linii i bez interpretacji tekstu podobnego do `[DONE]`. Następnie pobiera log wskazany przez `X-Run-Id`, sprawdza jego dane i porównuje sumę długości zapisów z liczbą odebranych bajtów treści odpowiedzi HTTP. W przypadku automatycznej dekompresji przez `requests` są to bajty po dekompresji; nie są to bajty pakietów transportowych.

To jest jawna obserwacja z logu serwera, **nie pasywna ekstrakcja zdarzeń SSE z sieci**. Surowa, nieescapowana serializacja fragmentów zawierających nowe linie nie umożliwia ogólnego i jednoznacznego odzyskania granic zapisu z samych linii. Kolektor nie próbuje przedstawiać takiego parsowania jako naprawionego.

Każdy nowy rekord archiwizuje wejściowy wektor długości, hash dokładnych bajtów logu i politykę detokenizacji. Wznowienie starych, niepoświadczonych rekordów w nowym protokole jest odrzucane. W konfiguracji znajdują się również hashe aktualnych plików lokalnych; pochodzenie faktycznie obsłużonego kodu serwera jest osobnym polem nowego logu aplikacji.

Domyślny backend metryki CLI zmieniono z `auto` na `tfidf`, czyli historyczną nazwę opcji dla niezważonego kosinusa częstości słów. Eliminuje to zależny od środowiska wybór MiniLM jako domyślnego φ. Samodzielny historyczny `client.py` ma ostrzeżenie, że jego wynik nie nadaje się do pomiarów badawczych; aktualna kampania go nie używa.

## Sprawdzenia i odtworzenie

Cztery testy obejmują podział odpowiedzi HTTP na fragmenty o rozmiarach 1, 2, 7 i 65536 bajtów, znaki CR/LF, pusty fragment, polskie znaki i emoji, dosłowny tekst `[DONE]` wewnątrz odpowiedzi, uszkodzone liczniki, niezgodny prompt/run ID oraz uciętą odpowiedź. Wszystkie przeszły. Test nie ładuje modeli i nie korzysta z sieci.

```bash
python3 -m unittest experiment_validation.tests.test_log_length_collector -v
python3 experiment_validation/revalidation_20261031/collection/rebuild_canonical_traces.py
```

Szczegóły sprawdzeń per model i hashe znajdują się w [canonical_trace_audit.json](canonical_trace_audit.json). Skrypt zapisuje ukończony plik wektorów atomowo. Surowe logi i historyczne rekonstrukcje pozostały niezmienione.
