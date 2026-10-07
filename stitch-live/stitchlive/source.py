"""Zrodla obrazow.

Dwa tryby:
  ZrodloKatalog  - obserwuje katalog w watku w tle, wrzuca nowe pliki do kolejki
  ZrodloWsadowe  - wczytuje istniejacy katalog naraz, do testow bez symulacji lotu
"""

from pathlib import Path
import logging
import queue
import threading
import time

log = logging.getLogger(__name__)

ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".tif", ".tiff"}


def lista_obrazow(katalog):
    """Posortowana lista plikow obrazow. Kolejnosc alfabetyczna odpowiada
    zwykle kolejnosci wykonania zdjec przez kapsule."""
    # pliki ukryte (np. .tmp_ w trakcie kopiowania) sa pomijane
    return sorted(p for p in Path(katalog).iterdir()
                  if p.suffix in ROZSZERZENIA and not p.name.startswith("."))


class ZrodloKatalog:
    """Obserwuje katalog i wrzuca nowe pliki do kolejki.

    Dziala w osobnym watku, wiec odczyt z dysku nie blokuje przetwarzania.
    """

    def __init__(self, katalog, kolejka, interwal_s=1.0, od_poczatku=False,
                 stabilizacja_s=0.4):
        self.katalog = Path(katalog)
        self.kolejka = kolejka
        self.interwal = interwal_s
        self.stabilizacja = stabilizacja_s
        # nazwy plikow juz wrzuconych do kolejki - kazdy trafia tylko raz
        self.widziane = set()
        # flaga zatrzymania - bezpieczna do ustawiania z innego watku
        self._stop = threading.Event()
        self._watek = None

        if not od_poczatku:
            # pliki juz obecne przy starcie sa pomijane
            self.widziane = {p.name for p in lista_obrazow(self.katalog)}
            log.info("Pomijam %d plikow obecnych przy starcie", len(self.widziane))

    def start(self):
        # Watek w tle. daemon=True: nie blokuje zamkniecia programu (np. po Ctrl+C).
        self.katalog.mkdir(parents=True, exist_ok=True)
        self._watek = threading.Thread(target=self._petla, daemon=True)
        self._watek.start()
        log.info("Obserwuje katalog: %s", self.katalog)

    def _petla(self):
        while not self._stop.is_set():
            try:
                for p in lista_obrazow(self.katalog):
                    if p.name in self.widziane:
                        continue
                    # plik moze byc jeszcze zapisywany - czekamy az rozmiar
                    # przestanie sie zmieniac
                    if not self._plik_gotowy(p):
                        continue
                    self.widziane.add(p.name)
                    self.kolejka.put(p)
            # katalog mogl zostac chwilowo usuniety - probujemy w nastepnym obiegu
            except FileNotFoundError:
                pass
            time.sleep(self.interwal)

    def _plik_gotowy(self, p):
        try:
            # Dwa odczyty rozmiaru w odstepie 'stabilizacja' sekund - jesli sie roznia,
            # plik jest wciaz zapisywany (np. kopiowany z drona).
            r1 = p.stat().st_size
            time.sleep(self.stabilizacja)
            return r1 > 0 and p.stat().st_size == r1
        except OSError:
            return False

    def stop(self):
        self._stop.set()
        if self._watek:
            # czekamy na zakonczenie watku, ale nie dluzej niz 2 s
            self._watek.join(timeout=2.0)


class ZrodloWsadowe:
    """Wczytuje caly katalog naraz. Do testow na materiale z wczesniejszego lotu."""

    def __init__(self, katalog, kolejka, opoznienie_s=0.0, limit=None):
        self.sciezki = lista_obrazow(katalog)
        if limit:
            self.sciezki = self.sciezki[:limit]
        self.kolejka = kolejka
        self.opoznienie = opoznienie_s
        self._watek = None

    def start(self):
        self._watek = threading.Thread(target=self._petla, daemon=True)
        self._watek.start()
        log.info("Tryb wsadowy: %d obrazow", len(self.sciezki))

    def _petla(self):
        for p in self.sciezki:
            self.kolejka.put(p)
            if self.opoznienie:
                time.sleep(self.opoznienie)
        # None w kolejce = koniec zdjec; petla glowna w cli.py konczy wtedy prace
        self.kolejka.put(None)          # sygnal konca

    def stop(self):
        pass
