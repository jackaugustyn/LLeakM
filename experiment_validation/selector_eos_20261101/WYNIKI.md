# Pełna kampania z poprawionym selektorem — ukończona

Wykonano i zweryfikowano **6600/6600** nowych rekonstrukcji. Wszystkie **3300/3300** jednokandydatowych predykcji MAP są identyczne z poprzednią kampanią.

Zmiana dotyczy sumowania wyników do pierwszego EOS włącznie; referencje, długości i seedy zostały zachowane. Wszystkich kandydatów i ich wyniki zapisano dla każdego segmentu.

| Kohorta | Model | Poprzednia średnia φ | Nowa średnia φ | Zmiana |
|---|---|---:|---:|---:|
| cuda_full | Qwen2.5-1.5B-Instruct | 0.535463 | 0.576845 | +0.041382 |
| cuda_full | Qwen2.5-3B-Instruct | 0.525143 | 0.557144 | +0.032001 |
| cuda_full | Llama-3.2-3B-Instruct | 0.514621 | 0.536473 | +0.021853 |
| cuda_full | Gemma-2-2B-it | 0.482001 | 0.509214 | +0.027213 |
| cuda_full | Phi-1.5 | 0.516368 | 0.542048 | +0.025681 |
| cuda_full | TinyLlama-1.1B-Chat | 0.079159 | 0.083221 | +0.004062 |
| cuda_full | Phi-3.5-mini-instruct | 0.121017 | 0.129744 | +0.008728 |
| fresh_tinyllama_cuda | TinyLlama-1.1B-Chat | 0.435205 | 0.467690 | +0.032485 |
| fresh_phi_cuda | Phi-3.5-mini-instruct | 0.504151 | 0.526942 | +0.022791 |

Archiwalne TinyLlama/Phi-3.5 nadal są diagnostyką referencji; oddzielnych świeżych kohort nie scala się z pozostałymi modelami w nowy ranking.

Kontrole: **4/6** kontrastów istotnych po Holm. Obrony: **10/10**.

Szczegółowe porównania sparowane, przedziały ufności i audyt wyboru kandydatów: `validation/audit_campaign.json`. Wszystkie liczby, metryki i protokoły: `completed_results.json`.

Ten raport opisuje ukończone obliczenia. Artykuł wymaga integracji tych wyników i kontroli nowego PDF; wcześniejszy artykuł opisuje poprzednią kampanię.
