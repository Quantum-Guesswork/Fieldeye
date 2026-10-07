# scripts/ — skrypty pomocnicze

| Skrypt | Do czego | Przykład |
|---|---|---|
| `download_datasets.sh` | pobiera prawdziwe zdjęcia z drona z repozytorium ODMdata | `./download_datasets.sh brighton_beach` |
| `make_test_data.py` | generuje sztuczne pole pocięte na kadry (starsza, prostsza wersja symulatora — nowsza to `python -m stitchlive.cli rozloz`) | `python make_test_data.py dane/test --rows 3 --cols 6` |
| `generuj_pole.py` | generuje realistyczne zdjęcia pola z drona (marchew, kukurydza, burak) z etykietami chwastów w formacie YOLO | `python generuj_pole.py dane_testowe --liczba 6 --uprawa mix` |
| `simulate_flight.sh` | kopiuje zdjęcia do katalogu w odstępach czasu, jak dron w locie | `./simulate_flight.sh zrodlo/ naplyw/ 2` |

## download_datasets.sh

```bash
./scripts/download_datasets.sh list              # lista zbiorów
./scripts/download_datasets.sh brighton_beach    # 18 zdjęć, ~62 MB
./scripts/download_datasets.sh mygla dane/       # do wskazanego katalogu
```

Wymaga `git`. Zdjęcia mają ~4000×3000 px — przy pierwszym składaniu użyj
`--limit` albo zmniejsz `--megapix`.

## simulate_flight.sh

Kopiuje przez plik tymczasowy i zmienia nazwę — operacja atomowa, więc
obserwujący program (`watch`) nigdy nie zobaczy pliku w połowie zapisu.

```bash
# terminal 1
python -m stitchlive.cli watch naplyw --renderuj-co 4 -o wyniki
# terminal 2
./scripts/simulate_flight.sh dane/test/images naplyw 1
```

Ten sam efekt daje `rozloz --na-zywo naplyw --odstep 1`, bez osobnego skryptu.

## generuj_pole.py

Rysuje pole proceduralnie: gleba z grudkami i wilgocią, ślady opon, rzędy
uprawy w trzech warstwach tonalnych (cień, środek, strona od słońca), linia
kroplująca, chwasty szerokolistne i trawiaste — w międzyrzędziach i w samych
rzędach.

```bash
python scripts/generuj_pole.py dane_testowe --liczba 6 --uprawa mix
python scripts/generuj_pole.py dane_testowe --uprawa kukurydza --chwasty 2.0
```

Wynik: `images/`, `labels/` (YOLO), `podglad/` (ramki), `classes.txt`, `opis.json`.
Jedno zdjęcie 3600×2200 generuje się ok. 10–15 s.
