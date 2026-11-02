# Artykuł oparty na zakończonych obliczeniach — 1 listopada 2026

Aktualny artykuł: [PDF](../../../article1/template.pdf) i [LaTeX](../../../article1/template.tex). PDF ma 21 stron. Rewizja obejmuje streszczenie, metody, wyniki, dyskusję, wnioski, wszystkie 12 tabel i wszystkie 4 rysunki.

Podstawą są zakończone i niezależnie sprawdzone **6600 rekonstrukcji T5 oraz 600 nowych generacji odpowiedzi**. W tej rewizji nie wykonywano kolejnej inferencji: wykresy wygenerowano bezpośrednio z końcowych danych, a wartości tabel ponownie porównano z niezależnymi rejestrami wyników. Poprawne wartości liczbowe poprzedniej integracji zachowano.

## Zmiany merytoryczne

- Streszczenie i wnioski eksponują kontrolowane porównanie detokenizacji: dla tych samych nowych token IDs stary dekoder odtwarza wszystkie 600 archiwalnych tekstów i wektorów długości. Zmiana obu elementów na poprawne dekodowanie daje przyrost średniego φ o **0,3560** dla TinyLlama i **0,3831** dla Phi-3.5. To wspólny efekt zmiany wejścia i referencji, bez rozdzielenia ich wkładu.
- Średnie nowych, poprawnie dekodowanych kohort wynoszą **0,4352** i **0,5042**. Zachowano ich oddzielność od pięciu pozostałych kohort archiwalnych. Archiwalne wyniki dwóch modeli pozostają diagnostyczne.
- Wnioski zachowują wynik **3/6** istotnych kontrastów kontroli i **10/10** kontrastów obron po korekcie Holma. Redukcje wyników zamrożonego rekonstruktora wynoszą **36,7–88,7%**. Wyników MAP nie interpretuje się jako odporności wobec dowolnego atakującego; test i kalibracja współdzielą scenariusze, a transformacje zmieniają segmentację i zakres wejścia T5.
- Metody wskazują, że nowa implementacja generuje całą sekwencję przed emisją fragmentów. Waliduje długości, a nie czasowe zachowanie generowania strumieniowego. Kontekstem następnego segmentu jest wyłącznie poprzedni wybrany segment.

## Wszystkie tabele

Sprawdzono **99 wierszy w 12 tabelach**. Dziesięć tabel ilościowych zawiera 423 kontrolowane pola liczbowe. Dwie opisowe tabele sprawdzono względem zachowanych metadanych i wszystkich 2100 rekordów źródłowych; nie jest to nowa zewnętrzna weryfikacja kart modeli.

| Tabela | Zakres i kontrola |
|---|---|
| 1 — modele | 7 checkpointów, opisy parametrów i rodziny tokenizerów; doprecyzowane ograniczenie historycznych rewizji |
| 2 — długości odpowiedzi | 7 modeli; tokeny, znaki, bajty/token i ucięcia; wyjaśnione uśrednianie po odpowiedziach |
| 3 — pokrycie enkodera | 6 warunków × 2 modele; doprecyzowany zamrożony atakujący i ponowne użycie wejść przez 5 ziaren |
| 4 — kontrole śladu | 8 warunków; średnie, przedziały i różnice |
| 5 — wyniki główne | 7 modeli; wszystkie metryki i LR; usunięte dwuznaczne „Complete reconstructions” |
| 6 — EOS | 7 modeli; liczności i wyniki podzbiorów; jawne procenty LR |
| 7 — niepewność statystyczna | 7 modeli; bootstrap promptów i przedziały Wilsona |
| 8 — scenariusze | 7 modeli; bootstrap 60 scenariuszy |
| 9 — nowe referencje | 2 modele; metryki, przedziały, LR i EOS; doprecyzowane limity T5 i znaczenie EOS |
| 10 — obrony | 12 warunków; średnie, przedziały, redukcje, rzeczywisty narzut i liczba jednostek; „Units” zastępuje mylące „Frames” |
| 11 — MAP | wszystkie 22 warunki; średnie, skrzyżowany bootstrap, odchylenia między ziarnami i redukcje |
| A1 — katalogi źródłowe | wszystkie 7 katalogów istnieje; każdy zawiera 300 zgodnych rekordów |

## Wszystkie rysunki

1. **Model zagrożenia:** przebudowany schemat rozdziela wejście atakującego od referencji używanej przez ewaluator. Nie utożsamia zapisów instrumentacji z poprawnymi zdarzeniami SSE ani rekordami TLS.
2. **Rekonstrukcja:** schemat pokazuje transformację i opcjonalną odwrotność MAP przed segmentacją, limit 80 tokenów, wybór kandydata na każdym segmencie i właściwą pętlę kontekstu.
3. **Wyniki modeli:** nowy wykres rozdziela archiwalne kohorty od sparowanego porównania sposobów dekodowania. Marginalne przedziały uwzględniają 60 scenariuszy; podpisane różnice mają sparowane przedziały scenariuszowe.
4. **Obrony:** nowy wykres pokazuje oba modele osobno na jednakowych osiach. Narzut pochodzi z faktycznie zapisanych przekształconych wektorów, również dla losowego paddingu.

Oba wykresy ilościowe mają eksporty PDF, SVG i PNG w `article1/figures/`. Ich [dane JSON](figure_data.json), pięć plików CSV, [generator](generate_figures.py) i [manifest](figure_manifest.json) umożliwiają odtworzenie i kontrolę 53 wartości wykresów bez inferencji modeli. Schematy pozostają w kodzie TikZ artykułu.

## Sprawdzenie i odtworzenie

- [Kontrola liczb i źródeł](manuscript_numeric_validation.json): **PASS — 476 wartości**, 12 tabel, wszystkie serie wykresów, kompletność warunków oraz powiązania SHA-256.
- [Przegląd schematów](conceptual_figure_review.json) i [przegląd wizualny tabel](table_visual_review.json): bez błędów merytorycznych lub układu wymagających dalszej korekty. Trzy końcowe sugestie dotyczące jednostek i EOS uwzględniono.
- [Końcowa kontrola dokumentu](final_build_validation.json): kompilacja poprawna, bez przepełnionych linii i nieokreślonych odwołań; sprawdzone obrazy stron, podpisy wszystkich tabel i rysunków oraz zachowanie cytowań i deklaracji.
- [Manifest bieżącej rewizji](revision_manifest.json) wiąże źródła, dokumenty, wyniki kontroli i gotowy PDF. Wcześniejsze manifesty eksperymentu i pierwszej integracji pozostają historycznymi zapisami.

Z katalogu głównego projektu:

```bash
python experiment_validation/revalidation_20261031/article_revision_latest_20261101/generate_figures.py
python experiment_validation/revalidation_20261031/article_revision_latest_20261101/check_manuscript_numbers.py
cd article1
latexmk -pdf -interaction=nonstopmode -halt-on-error template.tex
```

Rewizja zachowuje docelowy profil Electronics / Artificial Intelligence oraz wskazany Special Issue, autorów, bibliografię i deklaracje. Wyniki pozostają miarami podobieństwa leksykalnego w eksperymencie z instrumentacją; nie stanowią pomiaru semantycznego ujawnienia danych ani przechwytywania zaszyfrowanego ruchu. Publikacja kompletnego nowego pakietu danych pozostaje osobnym krokiem.
