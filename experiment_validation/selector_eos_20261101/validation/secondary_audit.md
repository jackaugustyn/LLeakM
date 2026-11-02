# Końcowa kontrola dodatkowych 3900 rekonstrukcji

Status: **PASS**. Zweryfikowano 240 kontroli, 360 prób obron i 3300 prób MAP. Nie wykonywano inferencji ani nie zmieniano poświadczonych artefaktów.

Ponownie obliczono wszystkie metryki tekstowe, narzuty, 42 średnie komórkowe, 78 przedziałów bootstrapu i 36 kontrastów. Największa różnica metryk: 0; przedziałów: 5.55e-17.

Kontrole: 4/6 istotnych porównań po Holm. Obrony: 10/10. Redukcje pilotowe: 43.7688–89.5798%.

Wszystkie 3300/3300 predykcji MAP odtworzono dokładnie względem wcześniejszego archiwum. Przepis bootstrapu pochodzi z kampanii 31 października i jest zachowany; jej predykcje oraz przedziały MAP powinny zostać odtworzone. Różnice względem starszych wrześniowych przedziałów wynikają z innego historycznego przepisu ziaren.

Dokładne liczby tabel, zakresy, dane wykresu i pokrycie enkodera: `secondary_audit.json`. Koszty losowego paddingu pochodzą z tych samych zamrożonych realizacji co 31 października; nie należy mieszać ich z wartościami oczekiwanymi.
