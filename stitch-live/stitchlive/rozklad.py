"""Rozklad jednego duzego zdjecia na kadry - symulacja lotu drona.

Bierzemy obraz calego pola (ortomozaike, zdjecie satelitarne, dowolne duze
zdjecie) i wycinamy z niego kadry tak, jakby robil je dron. Dzieki temu
znamy prawde - dokladne polozenie kazdego kadru - i mozemy sprawdzic,
czy skladanie ja odtwarza.

Dwa tryby:
  trasa   - przelot "kosiarka" (linie tam i z powrotem), z losowym
            odchyleniem pozycji, kursu i wysokosci. Najbardziej realistyczny.
  losowy  - losowe miejsca ciecia. Kazdy kolejny kadr lezy w losowym
            kierunku od poprzedniego, ale zawsze z zadanym minimalnym
            pokryciem, zeby dalo sie go dopasowac.

Zaklocenia, ktore mozna wlaczyc:
  obrot      - odchylenie kursu drona (stopnie)
  skala      - zmiana wysokosci lotu (kadr obejmuje wiekszy/mniejszy teren)
  ekspozycja - zmiana jasnosci miedzy kadrami
  rozmycie   - rozmycie ruchowe od drgan
  szum       - szum matrycy
"""

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
# Wszystkie parametry symulacji w jednym miejscu. @dataclass generuje
# konstruktor, wiec wystarczy Ustawienia(tryb="losowy", obrot=5) itd.
# Zapisywane w prawda.json (asdict), zeby wiadomo bylo, jak powstaly kadry.
class Ustawienia:
    tryb: str = "trasa"          # trasa | losowy
    kadr: int = 640              # rozmiar zapisanego kadru w px
    zasieg: float = 0.0          # ile px zrodla obejmuje kadr; 0 = tyle co kadr
    overlap: float = 0.65        # pokrycie wzdluz linii (trasa) / minimalne (losowy)
    sidelap: float = 0.5         # pokrycie miedzy liniami (trasa)
    liczba: int = 0              # ile kadrow (losowy); 0 = automatycznie
    jitter: float = 0.08         # odchylenie pozycji jako ulamek kroku
    obrot: float = 0.0           # odchylenie standardowe kursu w stopniach
    obrot_linii: bool = False    # co druga linia obrocona o 180 st. (jak prawdziwy dron)
    skala: float = 0.0           # odchylenie standardowe wysokosci (ulamek)
    ekspozycja: float = 0.04
    rozmycie: float = 0.0        # prawdopodobienstwo rozmycia ruchowego kadru
    szum: float = 0.0            # odchylenie standardowe szumu (poziomy jasnosci)
    winieta: float = 0.0         # przyciemnienie rogow kadru, 0-1 (typowe dla kamer dronowych)
    seed: int = 7


# ---------------- geometria wycinania ----------------

def macierz_kadru(cx, cy, kat_st, zasieg, kadr):
    """Macierz 2x3 przenoszaca piksel KADRU na piksel ZRODLA.

    Uzywana z flaga WARP_INVERSE_MAP: dla kazdego piksela kadru mowi,
    skad w zrodle go wziac. Obrot wokol srodka kadru, skala = zasieg/kadr.
    """
    # s - ile pikseli zrodla przypada na piksel kadru; a, b - skladowe obrotu ze skala.
    # Wyrazy wolne dobrane tak, zeby srodek kadru (h, h) trafil w (cx, cy) zrodla.
    s = zasieg / kadr
    t = math.radians(kat_st)
    a, b = s * math.cos(t), s * math.sin(t)
    h = kadr / 2.0
    return np.array([[a, -b, cx - a * h + b * h],
                     [b,  a, cy - b * h - a * h]], dtype=np.float64)


def naroza_kadru(M, kadr):
    """Polozenie naroznikow kadru w zrodle - do podgladu i oceny."""
    pts = np.array([[0, 0, 1], [kadr, 0, 1], [kadr, kadr, 1], [0, kadr, 1]], float)
    return (pts @ M.T).tolist()


# Wycina kadr. INTER_AREA - najlepsza interpolacja przy zmniejszaniu.
# BORDER_REFLECT - gdyby obrocony kadr wystawal poza zrodlo, rogi zostana
# wypelnione odbiciem obrazu zamiast czarnego tla.
def wytnij(zrodlo, M, kadr):
    return cv2.warpAffine(zrodlo, M, (kadr, kadr),
                          flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


# ---------------- planowanie polozen ----------------

def margines(zasieg, u):
    """Minimalna odleglosc srodka kadru od krawedzi zrodla.

    Bez obrotu wystarczy polowa kadru. Obrocony kadr siega dalej - o czynnik
    |cos| + |sin| kata. Zmiana wysokosci powieksza kadr. Liczymy realny
    najgorszy przypadek (3 odchylenia standardowe), a nie teoretyczne maksimum,
    ktore zabieraloby niepotrzebnie duzo miejsca na trase.
    """
    # realny najgorszy przypadek: 3 odchylenia standardowe, nie maksimum teoretyczne
    # katowi powyzej 45 st. nie ma sensu - |cos| + |sin| jest wtedy znow mniejsze
    kat = math.radians(min(45.0, 3 * u.obrot)) if u.obrot else 0.0
    z = zasieg * (min(1.33, 1 + 3 * u.skala) if u.skala else 1.0)
    return z / 2.0 * (abs(math.cos(kat)) + abs(math.sin(kat))) + 2

def plan_trasa(W, H, zasieg, u, rng):
    """Przelot tam i z powrotem po liniach poziomych."""
    # Krok miedzy kadrami z pokrycia: przy pokryciu 65% kolejny kadr przesuwa sie
    # o 35% swojej szerokosci.
    krok_x = zasieg * (1 - u.overlap)
    krok_y = zasieg * (1 - u.sidelap)
    marg = margines(zasieg, u)

    if W < 2 * marg or H < 2 * marg:
        raise ValueError(f"Obraz {W}x{H} za maly na kadr obejmujacy {zasieg:.0f} px")

    xs = np.arange(marg, W - marg + 1, krok_x)
    ys = np.arange(marg, H - marg + 1, krok_y)
    if len(xs) < 2:
        xs = np.array([marg, W - marg])

    plan = []
    for i, y in enumerate(ys):
        # co druga linia w przeciwna strone - dron zawraca na koncu pasa
        linia = xs if i % 2 == 0 else xs[::-1]
        kurs = 180.0 if (u.obrot_linii and i % 2 == 1) else 0.0
        for x in linia:
            # Plan to lista (x, y, kurs, numer_linii). Jitter - dron nigdy nie trafia
            # idealnie w zaplanowany punkt.
            plan.append((x + rng.normal(0, u.jitter * krok_x),
                         y + rng.normal(0, u.jitter * krok_y),
                         kurs, i))
    return plan


def plan_losowy(W, H, zasieg, u, rng):
    """Losowe miejsca ciecia z gwarancja pokrycia miedzy kolejnymi kadrami.

    Kolejny kadr lezy w losowym kierunku od poprzedniego, w odleglosci
    dajacej pokrycie co najmniej u.overlap. Gdy wypadlby poza obraz,
    losujemy kierunek ponownie.
    """
    marg = margines(zasieg, u)
    if W < 2 * marg or H < 2 * marg:
        raise ValueError(f"Obraz {W}x{H} za maly na kadr obejmujacy {zasieg:.0f} px")
    wolne = (W - 2 * marg) * (H - 2 * marg)
    if wolne < (zasieg * 2) ** 2:
        log.warning("Malo miejsca na trase (%.0fx%.0f px) - kadry beda sie "
                    "skupiac w jednym miejscu. Uzyj wiekszego obrazu albo "
                    "mniejszego --kadr / --obrot.", W - 2 * marg, H - 2 * marg)

    # Domyslna liczba kadrow: tyle, zeby obszar obrazu byl pokryty ok. 2,5 razy.
    n = u.liczba or max(8, int((W * H) / (zasieg * zasieg) * 2.5))
    # najdalszy krok, przy ktorym pokrycie z poprzednim kadrem jest jeszcze >= overlap
    max_krok = zasieg * (1 - u.overlap)

    x, y = rng.uniform(marg, W - marg), rng.uniform(marg, H - marg)
    plan = [(x, y, 0.0, 0)]
    kierunek = rng.uniform(0, 2 * math.pi)

    for _ in range(n - 1):
        for proba in range(50):
            # kierunek zmienia sie plynnie - jak dron, ktory skreca, a nie teleportuje sie
            kierunek += rng.normal(0, 0.35)
            # odleglosc 60-100% maksymalnej - zawsze z wymaganym pokryciem
            d = rng.uniform(0.6, 1.0) * max_krok
            nx, ny = x + d * math.cos(kierunek), y + d * math.sin(kierunek)
            if marg <= nx <= W - marg and marg <= ny <= H - marg:
                break
            kierunek += math.pi / 2          # odbicie od krawedzi
        else:
            # 50 prob bez skutku (zakleszczenie w rogu) - kadr w tym samym miejscu
            nx, ny = x, y
        x, y = nx, ny
        plan.append((x, y, 0.0, 0))
    return plan


# ---------------- zaklocenia ----------------

def _maska_winiety(h, w, sila):
    """Jasnosc malejaca od srodka ku rogom: 1 - sila * r^2.

    Uproszczenie prawdziwego spadku jasnosci obiektywu (w fizyce ok. cos^4
    kata padania); dla malych katow kwadrat odleglosci jest dobrym przyblizeniem.
    """
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot((x - w / 2) / (w / 2), (y - h / 2) / (h / 2)) / np.sqrt(2)   # 0 srodek, 1 rog
    return (1.0 - sila * r ** 2)[:, :, None]


# Naklada na wyciety kadr zaklocenia typowe dla kamery drona. Kolejnosc ma
# znaczenie: najpierw optyka (winieta), potem ekspozycja, ruch, na koncu szum
# matrycy - tak jak w prawdziwym torze obrazu.
def zaklocenia(img, u, rng):
    if u.winieta:
        img = np.clip(img.astype(np.float32) * _maska_winiety(*img.shape[:2], u.winieta),
                      0, 255).astype(np.uint8)
    if u.ekspozycja:
        # jeden losowy wspolczynnik jasnosci na caly kadr - automatyka ekspozycji
        g = 1.0 + rng.normal(0, u.ekspozycja)
        img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)
    if u.rozmycie and rng.random() < u.rozmycie:
        k = int(rng.integers(5, 13))
        # Jadro rozmycia ruchowego: pozioma linia jedynek (usrednianie k pikseli w poziomie).
        jadro = np.zeros((k, k), np.float32)
        jadro[k // 2, :] = 1.0 / k           # rozmycie wzdluz kierunku lotu
        img = cv2.filter2D(img, -1, jadro)
    if u.szum:
        # Szum niezalezny w kazdym pikselu i kanale - jak szum matrycy przy slabym swietle.
        img = np.clip(img.astype(np.float32) + rng.normal(0, u.szum, img.shape),
                      0, 255).astype(np.uint8)
    return img


# ---------------- glowna funkcja ----------------

def wczytaj_zrodlo(sciezka, max_wymiar=0):
    img = cv2.imread(str(sciezka), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Nie mozna wczytac obrazu: {sciezka}")
    if max_wymiar and max(img.shape[:2]) > max_wymiar:
        k = max_wymiar / max(img.shape[:2])
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        log.info("Zrodlo zmniejszone do %dx%d", img.shape[1], img.shape[0])
    return img


def rozloz(zrodlo, outdir, u=None, na_zywo=None, odstep=0.0):
    """Tnie obraz zrodlowy na kadry i zapisuje je razem z prawda.

    na_zywo: katalog, do ktorego kadry trafiaja stopniowo co `odstep`
             sekund - do testowania trybu watch.
    Zwraca slownik z opisem kadrow (to samo, co w prawda.json).
    """
    u = u or Ustawienia()
    # jedno ziarno na cala symulacje = te same kadry przy kazdym uruchomieniu
    rng = np.random.default_rng(u.seed)
    outdir = Path(outdir)
    imgdir = outdir / "images"
    if imgdir.exists():
        shutil.rmtree(imgdir)
    imgdir.mkdir(parents=True)

    H, W = zrodlo.shape[:2]
    zasieg = u.zasieg or u.kadr

    # najpierw plan (gdzie ciac), potem wlasciwe ciecie
    plan = (plan_losowy if u.tryb == "losowy" else plan_trasa)(W, H, zasieg, u, rng)
    marg = margines(zasieg, u)

    kadry = []
    podglad = zrodlo.copy()
    for i, (cx, cy, kurs, linia) in enumerate(plan, start=1):
        cx = float(np.clip(cx, marg, W - marg))
        cy = float(np.clip(cy, marg, H - marg))
        # kurs linii + losowe myszkowanie drona
        kat = kurs + (rng.normal(0, u.obrot) if u.obrot else 0.0)
        # Zmiana wysokosci = zmiana zasiegu kadru w terenie. Przycinamy do zakresu
        # uwzglednionego w marginesie, zeby kadr nie wyszedl poza obraz.
        z = zasieg * (1.0 + (rng.normal(0, u.skala) if u.skala else 0.0))
        z_max = min(1.33, 1 + 3 * u.skala) if u.skala else 1.0
        z = float(np.clip(z, zasieg * max(0.6, 2 - z_max), zasieg * z_max))

        M = macierz_kadru(cx, cy, kat, z, u.kadr)
        img = zaklocenia(wytnij(zrodlo, M, u.kadr), u, rng)

        nazwa = f"IMG_{i:04d}.jpg"
        cv2.imwrite(str(imgdir / nazwa), img, [cv2.IMWRITE_JPEG_QUALITY, 94])

        nar = naroza_kadru(M, u.kadr)
        # Prawda o kadrze: gdzie byl srodek, jaki kat i zasieg, gdzie narozniki.
        # Na tej podstawie ocena sprawdza, czy mozaika zgadza sie z oryginalem.
        kadry.append({"plik": nazwa, "srodek": [round(cx, 1), round(cy, 1)],
                      "kat_st": round(kat, 2), "zasieg_px": round(z, 1),
                      "linia": linia, "naroza": [[round(a, 1), round(b, 1)] for a, b in nar]})

        pts = np.array(nar, np.int32).reshape(-1, 1, 2)
        cv2.polylines(podglad, [pts], True, (0, 0, 255), 2)
        cv2.putText(podglad, str(i), (int(cx) - 10, int(cy) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # linia trasy - kolejnosc wykonywania zdjec
    trasa = np.array([k["srodek"] for k in kadry], np.int32).reshape(-1, 1, 2)
    cv2.polylines(podglad, [trasa], False, (0, 255, 255), 2)

    cv2.imwrite(str(outdir / "oryginal.jpg"), zrodlo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(outdir / "podglad_trasy.jpg"), podglad, [cv2.IMWRITE_JPEG_QUALITY, 85])

    prawda = {"zrodlo_px": [W, H], "ustawienia": asdict(u), "kadry": kadry}
    (outdir / "prawda.json").write_text(json.dumps(prawda, indent=1), encoding="utf-8")
    log.info("Rozlozono na %d kadrow -> %s", len(kadry), imgdir)

    if na_zywo:
        _wypuszczaj(imgdir, Path(na_zywo), odstep)
    return prawda


def _wypuszczaj(imgdir, cel, odstep):
    """Kopiuje kadry do katalogu obserwowanego, po jednym, jak w locie.

    Zapis do pliku tymczasowego i zmiana nazwy - obserwator nigdy nie
    zobaczy pliku w polowie zapisu.
    """
    cel.mkdir(parents=True, exist_ok=True)
    for p in sorted(imgdir.glob("*.jpg")):
        tmp = cel / f".tmp_{p.name}"
        shutil.copy(p, tmp)
        # zmiana nazwy jest atomowa - plik pojawia sie w calosci albo wcale
        tmp.rename(cel / p.name)
        log.info("  lot: %s", p.name)
        if odstep:
            time.sleep(odstep)
