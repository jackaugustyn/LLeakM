# Kampania EOS — uruchomiona

Uruchomiono pełną kolejkę **6600 nowych rekonstrukcji** na NVIDIA RTX 3070 Ti. Nadzorca zapisuje postęp niezależnie od bieżącej rozmowy.

- [Bieżący stan kolejki](queue/status.json)
- [Dokładny protokół](PROTOKOL.md)
- [Wynik kontroli uruchomienia](launch_validation.json)

Status można odczytać bez uruchamiania modeli:

```bash
python3 experiment_validation/selector_eos_20261101/status.py
```

Kolejka obejmuje dodatkowe 3900, główne 2100 i dwie świeże kohorty po 300 rekonstrukcji, a następnie niezależne audyty i zebranie wyników. Dwa pierwsze główne rekordy zostały już sprawdzone przed startem kolejki; są częścią 2100, nie dodatkowymi obserwacjami.

Dane w `preflight/` poza rzeczywistym głównym runem są testami przygotowawczymi i nie zwiększają liczebności kampanii. Wszystkie predykcje MAP muszą odtworzyć poprzednie wyniki.

Po ukończeniu wszystkich kontroli powstaną `completed_results.json`, `completed_manifest.json` i `WYNIKI.md`. Bieżący artykuł nadal opisuje wcześniejszą kampanię, do czasu integracji pełnych nowych wyników. Publiczny pakiet nie jest automatycznie wysyłany.
