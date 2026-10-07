# stitchlive

Łączenie zdjęć z przelotu drona w mozaikę — w czasie rzeczywistym, na bazie
OpenCV. Zawiera też **symulator lotu**: bierze jedno duże zdjęcie, tnie je na
kadry tak, jakby robił je dron (w różnych, także losowych miejscach), składa
z powrotem i porównuje wynik z oryginałem.

To jest **Program 1** z planu projektu — weryfikacja, czy materiał daje się
złożyć. Georeferencja i optymalizacja pamięci to Program 2.

---

## Szybki start

Polecenia `python -m stitchlive.cli ...` uruchamiaj **z katalogu projektu**.
Po `pip install -e .` możesz używać krótko `stitchlive ...` z dowolnego miejsca.

```bash
pip install -r requirements.txt

# rozłóż -> złóż -> oceń, na sztucznym polu (nic nie trzeba pobierać)
python -m stitchlive.cli demo

# to samo na własnym zdjęciu, cięcie losowe z zakłóceniami
python -m stitchlive.cli demo zdjecie.jpg --max-wymiar 2400 \
    --tryb-ciecia losowy --liczba 30 --obrot 6 --rozmycie 0.2 --szum 3
```

Wynik w katalogu `demo/`:

| Plik | Co to |
|---|---|
| `symulacja/podglad_trasy.jpg` | oryginał z zaznaczonymi kadrami i kolejnością lotu |
| `symulacja/images/` | wycięte kadry — tak, jakby zrobił je dron |
| `symulacja/prawda.json` | prawdziwe położenie, kąt i zasięg każdego kadru |
| `mozaika/mozaika_biezaca.jpg` | złożona mozaika |
| `ocena/ocena_nalozenie.jpg` | mozaika nałożona na oryginał |

---

## GUI (Streamlit)

```bash
streamlit run app.py
```

Otwiera się w przeglądarce, trzy zakładki:

| Zakładka | Co robi |
|---|---|
| **1. Rozkład zdjęcia** | wgrywasz zdjęcie (albo generujesz sztuczne pole), ustawiasz tryb lotu i zakłócenia, klikasz *Rozłóż* — widzisz trasę na oryginale i pierwsze kadry |
| **2. Kadry do pobrania** | galeria kadrów, podgląd pojedynczego z jego prawdziwym położeniem; pobieranie: same kadry (ZIP), kadry + `prawda.json` + podgląd trasy (ZIP), pojedynczy kadr |
| **3. Składanie** | kadry z zakładki 1 albo własne (zdjęcia lub ZIP); mozaika rośnie na żywo kadr po kadrze; ocena względem oryginału; pobieranie: mozaika (JPG), animacja składania (GIF), mozaika + kroki + raport (ZIP) |

Zestawy zakłóceń w zakładce 1:

| Zestaw | Obrót | Wysokość | Jasność | Rozmycie | Szum | Winieta |
|---|---|---|---|---|---|---|
| Czyste cięcie | 0 | 0 | 0 | 0 | 0 | 0 |
| Realistyczny dron | 3° | 3% | 5% | 10% kadrów | 2 | 0,30 |
| Trudne warunki | 6° | 5% | 7% | 25% kadrów | 4 | 0,35 |

„Trudne warunki" są na granicy możliwości metody — przy mocniejszych
zakłóceniach kadry zaczynają odpadać, co widać w metryce *W mozaice*.

**Winietowanie** (ciemniejsze rogi) jest typowe dla kamer dronowych i
przyciemnia akurat strefy nakładania kadrów. Metoda `global` wyrównuje
lokalnie kontrast (CLAHE) przed szukaniem punktów — bez tego przy winiecie
0,3 liczba dopasowanych par spadała ze 147 do 35.

---

## Symulator lotu — rozkład jednego zdjęcia

```bash
python -m stitchlive.cli rozloz [OBRAZ] [opcje] -o symulacja
```

Bez podanego obrazu generowane jest sztuczne pole (rzędy uprawy, chwasty,
kamienie).

### Dwa tryby cięcia

| `--tryb-ciecia` | Jak tnie |
|---|---|
| `trasa` (domyślnie) | przelot „kosiarką" — linie tam i z powrotem, z małym losowym odchyleniem pozycji |
| `losowy` | losowe miejsca cięcia; każdy kolejny kadr leży w losowym kierunku od poprzedniego, ale zawsze z zadanym minimalnym pokryciem, żeby dało się go dopasować |

### Zakłócenia

| Opcja | Symuluje | Przykład |
|---|---|---|
| `--obrot` | odchylenie kursu drona (odchylenie std. w stopniach) | `--obrot 5` |
| `--obrot-linii` | co druga linia obrócona o 180° (dron zawraca) | |
| `--skala` | zmiana wysokości lotu (ułamek) | `--skala 0.05` |
| `--ekspozycja` | zmiana jasności między kadrami | `--ekspozycja 0.06` |
| `--rozmycie` | prawdopodobieństwo rozmycia ruchowego kadru | `--rozmycie 0.2` |
| `--szum` | szum matrycy (poziomy jasności) | `--szum 3` |
| `--winieta` | ciemniejsze rogi kadru, 0–1 | `--winieta 0.3` |
| `--jitter` | losowe odchylenie pozycji (tryb trasa) | `--jitter 0.1` |

Pozostałe: `--kadr` (rozmiar kadru w px), `--zasieg` (ile px oryginału obejmuje
kadr), `--overlap`, `--sidelap`, `--liczba` (tryb losowy), `--seed`.

### Wypuszczanie kadrów jak w locie

```bash
# terminal 1 - składanie na bieżąco
python -m stitchlive.cli watch naplyw -m global --renderuj-co 4 -o wyniki

# terminal 2 - kadry trafiają do katalogu co 0,5 s
python -m stitchlive.cli rozloz zdjecie.jpg --tryb-ciecia losowy \
    --na-zywo naplyw --odstep 0.5
```

---

## Metody składania

### `global` — domyślna

Pary kadrów liczy OpenCV (ORB, test proporcji, RANSAC). Położenie **wszystkich**
kadrów naraz wyznacza jeden układ równań liniowych: przekształcenie
podobieństwa (obrót + skala + przesunięcie) jest liniowe w parametrach, więc
każda para punktów daje dwa równania. Kadr pierwszy jest układem odniesienia.
Błąd nie kumuluje się wzdłuż trasy.

**Dlaczego osobna metoda:** pipeline `cv2.detail` — i wzorcowy `cv2.Stitcher`
w trybie SCANS — przy kadrach obróconych o kilka stopni dają rozmytą mozaikę,
choć pojedyncze pary dopasowują się z błędem 0,1–0,8 px. Bez obrotu problemu nie
ma, bo same przesunięcia sumują się poprawnie w dowolnej kolejności.

Zabezpieczenia wbudowane w metodę:

- pary z za małym poparciem RANSAC albo absurdalną skalą są odrzucane,
- pary, które po rozwiązaniu nie pasują do reszty (błąd > 3 px), są usuwane
  i układ jest rozwiązywany ponownie,
- kadr niepołączony z resztą nie trafia do mozaiki, zamiast jej psuć;
  składana jest największa połączona grupa, nawet jeśli pierwszy kadr odpadł,
- rozmyty kadr, na którym domyślny detektor nic nie widzi, dostaje drugą próbę
  z niższym progiem.

### `detail` — pipeline `cv2.detail`

Model afiniczny (to samo co `cv2.Stitcher` SCANS, rozłożone na bloki).
Działa dobrze przy kadrach bez obrotu; ma okno przesuwne (`--okno N`)
dające stały czas dodania kadru.

### `basic` — `cv2.Stitcher`

Po każdym zdjęciu przelicza wszystko od nowa. Punkt odniesienia.

---

## Zmierzone wyniki

PSNR mozaiki względem oryginału (powyżej ~25 dB — bardzo dobra zgodność).

| Przypadek | `detail` / `cv2.Stitcher` | `global` |
|---|---|---|
| sztuczne pole, bez obrotu | 27,8 dB | 27,8 dB |
| sztuczne pole, obrót 8° | 19,7 dB (rozmyte) | 26,7 dB |
| prawdziwe zdjęcie z drona, trasa, 48 kadrów | — | 25,2 dB, 48/48 kadrów |
| prawdziwe zdjęcie, cięcie losowe + obrót 6° + zmiana wysokości + rozmycie + szum, 30 kadrów | — | 23,3 dB, 30/30 kadrów |

Prawdziwe zdjęcie: `DJI_0018.JPG` ze zbioru ODMdata *brighton_beach*
(las, droga, trawnik — drzewa to trudna, powtarzalna tekstura).

### Czas

48 kadrów, prawdziwe zdjęcie:

| Ustawienie | Czas |
|---|---|
| blokowe wyrównanie ekspozycji, 3000 cech | 117 s |
| wyrównanie per kadr, 1500 cech (domyślnie) | 10,5 s |

Jakość w obu przypadkach praktycznie ta sama (24,6 dB). Blokowe wyrównanie
skaluje się kwadratowo z liczbą nakładających się kadrów — zostało jako
`--bloki-ekspozycji`.

Tryb na żywo (16 kadrów co 0,5 s, render co 4 kadry): każde odświeżenie
mozaiki 0,1–0,45 s.

---

## Użycie

### `batch` — cały katalog naraz

```bash
python -m stitchlive.cli batch KATALOG -o wyniki
```

### `watch` — obserwacja katalogu w trakcie lotu

```bash
python -m stitchlive.cli watch KATALOG --renderuj-co 4 -o wyniki
```

Wątek w tle obserwuje katalog, wątek główny składa. Plik jest brany dopiero,
gdy przestanie rosnąć — inaczej program złapałby zdjęcie w trakcie zapisu.
Po zakończeniu strumienia mozaika jest renderowana jeszcze raz, żeby objąć
ostatnie kadry.

### `ocen` — porównanie mozaiki z oryginałem

```bash
python -m stitchlive.cli ocen mozaika.jpg oryginal.jpg --prawda prawda.json
```

Mozaika ma własny układ współrzędnych, więc najpierw jest dopasowywana do
oryginału, potem liczone są: skala, pokrycie, średni błąd jasności i PSNR.

### Najważniejsze opcje składania

| Opcja | Opis |
|---|---|
| `-m, --metoda` | `global` (domyślnie), `detail`, `basic` |
| `--megapix` | rozdzielczość robocza (domyślnie 0,6 Mpx) |
| `--zakres-par N` | dopasowuj kadr tylko z N poprzednimi; 0 = z każdym |
| `--renderuj-co N` | odświeżaj mozaikę co N kadrów |
| `--cechy` | liczba punktów na kadr (domyślnie 1500) |
| `--szwy` | wyznaczanie szwów — usuwa widoczne granice, wolniej |
| `--bloki-ekspozycji` | blokowe wyrównanie jasności — wolno |
| `--bez-ekspozycji` | bez wyrównania jasności |
| `--detektor` | `ORB` (domyślnie) albo `SIFT` |

Dla przelotu „kosiarką" `--zakres-par` warto ustawić na co najmniej dwie
długości linii — kadr z sąsiedniej linii leży w kolejności daleko.

---

## Wygenerowane zdjęcia pola

```bash
python scripts/generuj_pole.py dane_testowe --liczba 6 --uprawa mix
```

Realistyczne (proceduralne) zdjęcia pola z drona — marchew, kukurydza,
burak — razem z etykietami chwastów w formacie YOLO. Można je pociąć
symulatorem i złożyć z powrotem albo użyć do sprawdzenia potoku YOLO.
Szczegóły: `scripts/README.md`.

---

## Prawdziwe dane

```bash
./scripts/download_datasets.sh list
./scripts/download_datasets.sh brighton_beach
python -m stitchlive.cli batch dane/brighton_beach/images --limit 10 -o wyniki
```

Zbiory z ODMdata mają zdjęcia ~4000×3000 px — przy pierwszym uruchomieniu
użyj `--limit`.

---

## Testy

```bash
pip install pytest
python -m pytest tests/ -q
```

50 testów, m.in.: składanie kadrów obróconych, dokładność par (< 2 px),
kadr odniesienia = tożsamość, cięcie losowe z zakłóceniami, ratowanie
rozmytych kadrów, odrzucanie obcego kadru, tryb na żywo, ocena, CLI,
generator pola (poprawność etykiet YOLO, ramki trafiające w rośliny).

---

## Struktura

```
stitchlive/
├── globalny.py   metoda global - pary OpenCV + globalne wyrównanie
├── detail.py     pipeline cv2.detail, model afiniczny, okno przesuwne
├── basic.py      opakowanie cv2.Stitcher
├── rozklad.py    symulator lotu - cięcie jednego zdjęcia na kadry
├── ocena.py      porównanie mozaiki z oryginałem
├── pole.py       generator sztucznego pola
├── source.py     źródła obrazów (katalog obserwowany, tryb wsadowy)
└── cli.py        wiersz poleceń
app.py            GUI w Streamlit
scripts/generuj_pole.py  realistyczne zdjęcia pola z etykietami YOLO
```

---

## Ograniczenia

- **Brak georeferencji** — mozaika jest w układzie pierwszego kadru, bez
  współrzędnych geograficznych. To zadanie Programu 2 (GNSS/IMU).
- **Płaski teren, zdjęcia z nadiru** — model podobieństwa nie uwzględnia
  perspektywy ani różnic wysokości terenu.
- **Pamięć** — wszystkie kadry robocze są trzymane w pamięci; przy bardzo
  długich misjach trzeba obniżyć `--megapix`.
- **Czas dopasowania rośnie z liczbą kadrów**, jeśli `--zakres-par` = 0.
- **Plamy jasności** na jednolitych powierzchniach (asfalt) przy wyrównaniu
  per kadr — pomaga `--szwy` albo `--bloki-ekspozycji`.
