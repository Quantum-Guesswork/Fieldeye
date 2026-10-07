# tests/ — testy automatyczne

```bash
pip install pytest
python -m pytest tests/ -q          # ~1,5 minuty, 50 testów
python -m pytest tests/test_globalny.py -v
```

Uruchamiaj z katalogu głównego projektu.

| Plik | Co sprawdza |
|---|---|
| `test_globalny.py` | metoda `global`: składanie kadrów obróconych, dokładność par (< 2 px), kadr odniesienia = tożsamość, cięcie losowe z zakłóceniami, ratowanie rozmytych kadrów, odrzucenie obcego kadru, zakres par, końcowe renderowanie |
| `test_rozklad.py` | symulator lotu i ocena: pokrycie w trybie losowym, powtarzalność przy tym samym ziarnie, za mały obraz, kursy 180°, tryb na żywo, progi werdyktu, komendy `rozloz` i `demo` |
| `test_generuj_pole.py` | generator pola: rozmiar i typ obrazu, obie klasy chwastów, ramki w granicach obrazu, powtarzalność przy tym samym ziarnie, środek ramki trafia w roślinę, format YOLO i jego odwracalność na piksele, wywołanie z wiersza poleceń |
| `test_stitch.py` | metody `basic` i `detail`, okno przesuwne, zwalnianie pamięci, CLI |

## Zasada testów

Kadry powstają przez pocięcie **znanego** obrazu, więc wiadomo, jak powinna
wyglądać mozaika. Testy nie sprawdzają tylko „czy kod się wykonał", ale czy
wynik zgadza się z prawdą — np. PSNR powyżej progu albo mediana błędu par
poniżej 2 pikseli.
