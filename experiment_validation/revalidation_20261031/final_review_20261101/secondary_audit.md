# Końcowa kontrola dodatkowych 3900 rekonstrukcji

Status: **PASS**. Zweryfikowano 240 kontroli, 360 prób obron i 3300 prób MAP. Nie wykonywano inferencji ani nie zmieniano poświadczonych artefaktów.

Ponownie obliczono wszystkie metryki tekstowe, narzuty, 42 średnie komórkowe, 78 przedziałów bootstrapu i 36 kontrastów. Największa różnica metryk: 0; przedziałów: 5.55e-17.

Kontrole: 3/6 istotnych porównań po Holm. Obrony: 10/10. Redukcje pilotowe: 36.7031–88.7003%.

Wszystkie 3300/3300 predykcji MAP odtworzono dokładnie względem archiwum. Odmienione przedziały wynikają z nowego jawnego przepisu bootstrapu, nie zmiany predykcji.

Dokładne liczby tabel, zakresy, dane wykresu i pokrycie enkodera: `secondary_audit.json`. Średnie koszty losowego paddingu są zarejestrowanymi nowymi realizacjami; nie należy mieszać ich z wcześniej obliczonymi wartościami oczekiwanymi.
