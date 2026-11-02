# Stałe sekwencje ID: porównanie dawnej i pełnej detokenizacji

Rekonstrukcja leksykalna nie była tutaj ponownie uruchamiana. Użyto dokładnie tych samych, zapisanych ID z nowych generacji i przypiętych tokenizerów. Dawna funkcja dekodowała każdy token oddzielnie z `skip_special_tokens=False` i `clean_up_tokenization_spaces=False`.

| Model | n | Dawny replay = historyczny tekst | Dawny replay = historyczne fragmenty | Dawny replay = historyczne długości | Dawny replay = pełna nowa detokenizacja | Bez spacji: dawny replay / nowy tekst / archiwum |
|---|---:|---:|---:|---:|---:|---:|
| tinyllama_1_1b | 300 | 300 | 300 | 300 | 0 | 299 / 0 / 299 |
| phi_3_5_mini | 300 | 300 | 300 | 300 | 0 | 162 / 0 / 162 |

W obrębie zapisanej nowej sekwencji ID kontrast izoluje wpływ funkcji dekodującej na tekst i wektor bajtów. Zgodność z archiwalnym tekstem, fragmentami lub długościami nie odzyskuje nieobecnych historycznych ID i nie poświadcza historycznej wersji checkpointu. Różnicy wyników T5 między kampaniami nie przypisuje się na tej podstawie wyłącznie detokenizacji.
