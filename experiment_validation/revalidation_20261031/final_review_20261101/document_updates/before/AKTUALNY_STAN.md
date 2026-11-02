# Wznowione obliczenia — 31 października 2026

**Użytkownik wznowił pracę poleceniem „wznawiaj obliczenia”. Poprzednia pauza już nie obowiązuje.**

Stan opisowy zapisano 2026-10-31T14:58:14.155532+00:00; bieżące liczby czytać z plików statusu.

## Procesy i kolejka

- GPU: NVIDIA GeForce RTX 3070 Ti Laptop, 8 GiB; PyTorch 2.5.1+cu124 widzi CUDA.
- Główny T5: PID 187555, sesja narzędziowa 63692. W chwili zapisu 83/2100; status `running`, complete=false. Szacunek z początkowych próbek około 7 godzin dla samej kampanii głównej, nie dla całej kolejki.
- Nadzorca: PID 194635, osobna sesja procesu; log `queue/supervisor.log`, stan `queue/status.json`.
- Plan: `queue_plan.json`, SHA256 `b964254e282fecfcfb62d992411084eb655405050049c7e3d41464ed0c68917f`. Kodu objętego source_sha256 nie zmieniać podczas kolejki: kontrola wykryje zmianę i zatrzyma dalsze etapy.
- T5 source SHA256: `55ebb745e9cafd08d89931f17ef26a5858ff8884c526e16aa2b0201784c83dc1`.
- Główny protocol_id: `d5394a6ade10b4b46c841d5c69660674c7ada72079094036e1e891f148b8e924`.

Kolejność 12 etapów: główne T5 2100 → niezależna weryfikacja → kontrola/obrony/MAP 3900 → ponowna walidacja tych rekordów bez GPU → kontrola wag ofiar → TinyLlama 300 nowych generacji → ich T5 → weryfikacja → Phi-3.5 300 nowych generacji → ich T5 → weryfikacja → raport i manifest.

Modele działają kolejno, więc nie konkurują o VRAM. Phi używa FP16, bez kwantyzacji, z budżetem wag GPU 6 GiB i resztą w RAM. Wydajność i pełna generacja Phi pozostają do sprawdzenia podczas jej etapu. Pełna kolejka potrwa dłużej niż główne T5.

Kolejka weryfikuje kody wyjścia i kompletność, zapisuje stan co 15 sekund, nie ponawia automatycznie nieudanego etapu. Etapy inferencji mają limit 24 godzin każdy, weryfikacje 2 godziny, pobieranie wag 6 godzin. Osierocony aktywny proces blokuje powtórne uruchomienie tego samego etapu. SIGTERM nadzorcy zatrzymuje jego własny aktywny podproces; przy dołączonym głównym T5 ten zewnętrzny proces pozostaje uruchomiony.

## Zweryfikowane przygotowanie

- 23 testy napraw aparatury/statystyk przeszły.
- 5 testów niezależnego weryfikatora przeszło; rzeczywisty snapshot 41 rekordów CUDA przeszedł kontrolę logów, wejść enkodera i metryk (0 wcześniejszych zatrzymań, 38 obciętych wywołań enkodera). To nie jest kontrola kompletnej kampanii.
- Obie ofiary pobrane: 15/15 plików zgodnych, w tym hashe LFS wag. Rewizje są przypięte obecnie, ale ich tożsamość z historycznymi wagami nie jest ustalona.
- 4 testy nowego runnera generacji oraz 8 przypadków rzeczywistych tokenizerów przeszły. Modelowej generacji ofiar jeszcze nie wykonano.
- Plan dodatkowych eksperymentów: 240 kontroli, 360 obron i 3300 MAP; 660 zweryfikowanych logów, 3300 zgodnych kluczy i ziaren MAP; 8 testów i niezależny przegląd oraz 4 przypadki logiki zatrzymania przeszły.
- 5 kontroli nadzorcy (sukces, niekompletność, błąd procesu, osierocony proces, SIGTERM) i 2 kontrole zbierania raportu przeszły.

Raporty: `inference_validation/validation_receipt.json`, `fresh_references/readiness.json`, `fresh_references/cache_validation.json`, `secondary/READY.json`, `secondary/independent_review.json`, `queue_validation.json`, `collection_validation.json`.

## Artykuł

Dokończono aktualizację odnoszącą się do ukończonego przeliczenia archiwalnych statystyk: analiza 60 scenariuszy, 7 CI, te same 17/21 istotnych par; narzut losowego paddingu opisany jako oczekiwany 79,8% / 83,1%, wykres 81,4%. Rozróżniono historyczne błędy i aktualne naprawy kodu.

`article1/template.pdf` przebudowany: 20 stron, bez błędów, undefined references i overfull; sprawdzono wizualnie strony 13, 15 i 16. `article_updates/build_validation.json` wiąże PDF i TeX z tą wersją. Częściowe wyniki GPU nie zostały wstawione do artykułu.

## Po zakończeniu kolejki

`completed_results.json`, `completed_manifest.json` i `WYNIKI_WZNOWIENIA.md` powstaną wyłącznie po przejściu wszystkich etapów. Należy wtedy przejrzeć pełne wyniki i uzgodnić artykuł; nie łączyć automatycznie nowych referencji z historyczną tabelą siedmiu modeli. Dwie serie archiwalne nadal są diagnostyczne. Nowe generacje mają zweryfikowane token IDs i pełne dekodowanie, lecz inny jawny protokół/wersje modeli. Nie przypisywać całej zmiany wyniku wyłącznie naprawie kolektora lub detokenizacji.

Nie uruchamiać drugiej kopii kampanii. Najpierw sprawdzić PID, `queue/status.json` i odpowiedni status etapu. Po awarii należy zachować zapisane wyniki i diagnozę; nowy PID dołączonego T5 wymaga nowego planu/katalogu nadzorcy, ponieważ istniejący plan jest związany hashem.
