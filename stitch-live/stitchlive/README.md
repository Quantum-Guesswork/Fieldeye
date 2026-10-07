# stitchlive/ — kod programu

Pakiet Pythona z całą logiką. GUI (`../app.py`) i wiersz poleceń (`cli.py`)
tylko go wywołują.

## Moduły

| Plik | Rola | Najważniejsze elementy |
|---|---|---|
| `globalny.py` | **domyślna metoda składania** | `SkladaczGlobalny` — pary kadrów z OpenCV (ORB + RANSAC), położenie wszystkich kadrów z jednego układu równań liniowych |
| `detail.py` | metoda `detail` | `SkladaczDetail` — pipeline `cv2.detail`, model afiniczny, okno przesuwne |
| `basic.py` | metoda `basic` | `SkladaczPodstawowy` — opakowanie `cv2.Stitcher` w trybie SCANS |
| `rozklad.py` | symulator lotu | `Ustawienia`, `rozloz()` — cięcie jednego zdjęcia na kadry, trasa lub losowo, z zakłóceniami |
| `ocena.py` | ocena wyniku | `ocen()` — dopasowuje mozaikę do oryginału i liczy PSNR, pokrycie, skalę |
| `pole.py` | dane testowe | `zbuduj_pole()` — sztuczne pole z rzędami, chwastami i kamieniami |
| `source.py` | źródła zdjęć | `ZrodloKatalog` (obserwuje katalog w wątku), `ZrodloWsadowe` |
| `cli.py` | wiersz poleceń | komendy `batch`, `watch`, `info`, `rozloz`, `ocen`, `demo` |

## Wspólny interfejs składaczy

Każdy składacz ma te same metody, więc GUI i CLI mogą je podmieniać:

```python
s = SkladaczGlobalny(work_megapix=0.6, renderuj_co=3)
for sciezka in zdjecia:
    if s.dodaj(sciezka):        # True = mozaika zaktualizowana
        podglad(s.mozaika)
s.zakoncz()                     # tylko global: końcowe renderowanie
s.raport_czasu()                # [{"zdjec": n, "czas_s": t, "ok": bool}, ...]
```

## Przepływ danych

```
rozklad.rozloz()  ->  images/*.jpg + prawda.json
                          |
source.Zrodlo*    ->  kolejka  ->  Skladacz*.dodaj()  ->  mozaika
                                                            |
                                  ocena.ocen(mozaika, oryginał, prawda)
```

Szczegóły algorytmów: dokument Word „Technologia i kod" oraz README
w katalogu głównym.
