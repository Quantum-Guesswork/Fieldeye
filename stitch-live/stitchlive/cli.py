"""Petla glowna i interfejs wiersza polecen."""

import argparse
import json
import logging
import queue
import sys
import time
from pathlib import Path

import cv2

from .source import ZrodloKatalog, ZrodloWsadowe, lista_obrazow
from .basic import SkladaczPodstawowy
from .detail import SkladaczDetail
from .globalny import SkladaczGlobalny
from .rozklad import Ustawienia, rozloz, wczytaj_zrodlo
from .ocena import ocen
from .pole import zbuduj_pole

log = logging.getLogger("stitchlive")


# Tworzy skladacz wybranej metody z parametrow wiersza polecen (--metoda).
def zbuduj_skladacz(args):
    if args.metoda == "basic":
        return SkladaczPodstawowy(downscale=args.downscale, tryb=args.tryb)
    if args.metoda == "global":
        return SkladaczGlobalny(
            work_megapix=args.megapix,
            zakres_par=args.zakres_par,
            detektor=args.detektor,
            wyrownaj_ekspozycje=not args.bez_ekspozycji,
            szwy=args.szwy,
            renderuj_co=args.renderuj_co,
            cechy=args.cechy,
            bloki_ekspozycji=args.bloki_ekspozycji,
        )
    return SkladaczDetail(
        okno=args.okno,
        work_megapix=args.megapix,
        conf_thresh=args.conf,
        detektor=args.detektor,
        model=args.model,
        wyrownaj_ekspozycje=not args.bez_ekspozycji,
        szwy=args.szwy,
        renderuj_co=args.renderuj_co,
        zakres_par=args.zakres_par,
    )


# Petla glowna - wspolna dla trybu wsadowego i obserwacji katalogu.
# Zrodlo (watek w tle) wrzuca sciezki do kolejki, a ta petla je pobiera
# i przekazuje skladaczowi. Po kazdym udanym odswiezeniu zapisuje mozaike.
def petla(kolejka, skladacz, outdir, zapisuj_kroki=False, timeout=None):
    """Pobiera zdjecia z kolejki i aktualizuje mozaike."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    t_start = time.time()

    while True:
        try:
            # get() czeka, az cos pojawi sie w kolejce. Z timeoutem: po N sekundach
            # bez nowych zdjec rzuca queue.Empty - koniec lotu.
            sciezka = kolejka.get(timeout=timeout) if timeout else kolejka.get()
        except queue.Empty:
            log.info("Brak nowych zdjec przez %.0f s - koncze", timeout)
            break

        if sciezka is None:            # sygnal konca z trybu wsadowego
            break

        n += 1
        log.info("[%d] %s", n, sciezka.name)

        try:
            # Blad jednego zdjecia (np. uszkodzony plik) nie przerywa calego lotu.
            ok = skladacz.dodaj(sciezka)
        except Exception as e:
            log.error("Blad przy %s: %s", sciezka.name, e)
            continue

        if ok and skladacz.mozaika is not None:
            cv2.imwrite(str(outdir / "mozaika_biezaca.jpg"), skladacz.mozaika,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            if zapisuj_kroki:
                cv2.imwrite(str(outdir / f"krok_{n:04d}.jpg"), skladacz.mozaika,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

    # koncowe renderowanie - np. gdy ostatnie kadry odpadly i licznik nie doszedl do progu
    if hasattr(skladacz, "zakoncz") and skladacz.zakoncz() and skladacz.mozaika is not None:
        cv2.imwrite(str(outdir / "mozaika_biezaca.jpg"), skladacz.mozaika,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    czas = time.time() - t_start
    raport = {
        "przetworzonych": n,
        "czas_calkowity_s": round(czas, 1),
        "mozaika": (list(skladacz.mozaika.shape[:2][::-1])
                    if skladacz.mozaika is not None else None),
        "historia_czasow": skladacz.raport_czasu(),
        "wyrownanie": skladacz.raport() if hasattr(skladacz, "raport") else None,
    }
    (outdir / "raport.json").write_text(
        json.dumps(raport, indent=2), encoding="utf-8")

    print()
    if skladacz.mozaika is not None:
        h, w = skladacz.mozaika.shape[:2]
        print(f"Mozaika: {w} x {h} px")
        print(f"Plik:    {outdir/'mozaika_biezaca.jpg'}")
    else:
        print("Nie udalo sie zlozyc mozaiki - zobacz logi powyzej")
    print(f"Zdjec:   {n}, czas: {czas:.1f} s")
    print(f"Raport:  {outdir/'raport.json'}")
    return raport


# Ponizej funkcje cmd_* - po jednej na komende (batch, watch, info, rozloz,
# ocen, demo). Wybiera je argparse na podstawie pierwszego argumentu.
def cmd_batch(args):
    """Tryb wsadowy - caly katalog naraz, bez symulacji czasu."""
    kolejka = queue.Queue()
    zrodlo = ZrodloWsadowe(args.katalog, kolejka,
                           opoznienie_s=args.opoznienie, limit=args.limit)
    skladacz = zbuduj_skladacz(args)
    zrodlo.start()
    petla(kolejka, skladacz, args.output, zapisuj_kroki=args.kroki)
    return 0


def cmd_watch(args):
    """Tryb obserwacji katalogu - praca w trakcie lotu."""
    kolejka = queue.Queue()
    zrodlo = ZrodloKatalog(args.katalog, kolejka,
                           interwal_s=args.interwal,
                           od_poczatku=args.od_poczatku)
    skladacz = zbuduj_skladacz(args)
    zrodlo.start()
    print(f"Obserwuje {args.katalog}. Ctrl+C konczy.")
    try:
        petla(kolejka, skladacz, args.output, zapisuj_kroki=args.kroki,
              timeout=args.timeout)
    except KeyboardInterrupt:
        print("\nPrzerwano.")
    finally:
        zrodlo.stop()
    return 0


def cmd_info(args):
    """Podstawowe informacje o zbiorze zdjec."""
    sciezki = lista_obrazow(args.katalog)
    if not sciezki:
        print("Brak obrazow w katalogu")
        return 1
    img = cv2.imread(str(sciezki[0]))
    print(f"Obrazow:    {len(sciezki)}")
    print(f"Rozmiar:    {img.shape[1]} x {img.shape[0]} px")
    print(f"Pierwszy:   {sciezki[0].name}")
    print(f"Ostatni:    {sciezki[-1].name}")
    laczny = sum(p.stat().st_size for p in sciezki) / 1e6
    print(f"Rozmiar:    {laczny:.0f} MB")
    return 0


# obraz zrodlowy dla symulatora: podany plik albo wygenerowane sztuczne pole
def _zrodlo(args):
    """Obraz podany przez uzytkownika albo wygenerowane sztuczne pole."""
    if args.obraz:
        return wczytaj_zrodlo(args.obraz, args.max_wymiar)
    log.info("Brak obrazu - generuje sztuczne pole %dx%d", args.pole_w, args.pole_h)
    return zbuduj_pole(args.pole_w, args.pole_h, seed=args.seed)


# argumenty wiersza polecen -> obiekt Ustawienia symulatora
def _ustawienia(args):
    return Ustawienia(
        tryb=args.tryb_ciecia, kadr=args.kadr, zasieg=args.zasieg,
        overlap=args.overlap, sidelap=args.sidelap, liczba=args.liczba,
        jitter=args.jitter, obrot=args.obrot, obrot_linii=args.obrot_linii,
        skala=args.skala, ekspozycja=args.ekspozycja, rozmycie=args.rozmycie,
        szum=args.szum, winieta=args.winieta, seed=args.seed)


def cmd_rozloz(args):
    """Tnie jedno zdjecie na kadry - symulacja robienia zdjec przez drona."""
    zrodlo = _zrodlo(args)
    prawda = rozloz(zrodlo, args.output, _ustawienia(args),
                    na_zywo=args.na_zywo, odstep=args.odstep)
    out = Path(args.output)
    print(f"Kadrow: {len(prawda['kadry'])}  ({args.kadr}x{args.kadr} px)")
    print(f"Zdjecia:        {out/'images'}")
    print(f"Podglad trasy:  {out/'podglad_trasy.jpg'}")
    print(f"Prawda:         {out/'prawda.json'}")
    return 0


def cmd_ocen(args):
    """Porownuje zlozona mozaike z oryginalem."""
    moz = cv2.imread(str(args.mozaika))
    ory = cv2.imread(str(args.oryginal))
    if moz is None or ory is None:
        print("Nie mozna wczytac mozaiki albo oryginalu")
        return 1
    prawda = json.loads(Path(args.prawda).read_text()) if args.prawda else None
    w = ocen(moz, ory, prawda, outdir=args.output)
    _drukuj_ocene(w)
    return 0 if w.get("ok") else 1


def _drukuj_ocene(w):
    print()
    if not w.get("ok"):
        print(f"Ocena nieudana: {w.get('powod')}")
        return
    print(f"Zgodne pary:   {w['zgodne_pary']}")
    print(f"Skala:         {w['skala']}   (1.0 = zgodna z oryginalem)")
    print(f"Pokrycie:      {w['pokrycie']:.1%}  obszaru przeleconego")
    print(f"Blad sredni:   {w['blad_sredni']}  (poziomy jasnosci 0-255)")
    print(f"PSNR:          {w['psnr_db']} dB")
    print(f"Werdykt:       {w['werdykt']}")


# Demo = trzy kroki naraz: rozloz -> zloz -> ocen. Najszybszy sposob,
# zeby sprawdzic caly program jednym poleceniem.
def cmd_demo(args):
    """Rozklad -> skladanie -> ocena, jednym poleceniem."""
    out = Path(args.output)
    zrodlo = _zrodlo(args)
    print("== 1/3 rozklad zdjecia na kadry ==")
    prawda = rozloz(zrodlo, out / "symulacja", _ustawienia(args))
    print(f"Kadrow: {len(prawda['kadry'])}")

    print("\n== 2/3 skladanie ==")
    kolejka = queue.Queue()
    ZrodloWsadowe(out / "symulacja" / "images", kolejka).start()
    if args.renderuj_co == 1:
        # w demo liczy sie wynik koncowy - renderujemy raz, po ostatnim kadrze
        args.renderuj_co = len(prawda["kadry"])
    skladacz = zbuduj_skladacz(args)
    petla(kolejka, skladacz, out / "mozaika")
    if skladacz.mozaika is None:
        return 1

    print("\n== 3/3 ocena wzgledem oryginalu ==")
    w = ocen(skladacz.mozaika, zrodlo, prawda, outdir=out / "ocena")
    _drukuj_ocene(w)
    print(f"\nPodglad trasy:  {out/'symulacja'/'podglad_trasy.jpg'}")
    print(f"Mozaika:        {out/'mozaika'/'mozaika_biezaca.jpg'}")
    print(f"Nalozenie:      {out/'ocena'/'ocena_nalozenie.jpg'}")
    return 0 if w.get("ok") else 1


def build_parser():
    ap = argparse.ArgumentParser(
        prog="stitchlive",
        description="Skladanie zdjec z drona w czasie rzeczywistym (OpenCV)")
    ap.add_argument("-v", "--verbose", action="store_true")
    # kazda komenda to osobny 'podparser' z wlasnymi opcjami
    sub = ap.add_subparsers(dest="cmd", required=True)

    # opcje skladania - wspolne dla batch, watch i demo
    def wspolne(p):
        p.add_argument("katalog", help="katalog ze zdjeciami")
        p.add_argument("-o", "--output", default="wyniki")
        wspolne_bez_katalogu(p)

    def wspolne_bez_katalogu(p):
        p.add_argument("-m", "--metoda", default="global",
                       choices=["global", "detail", "basic"],
                       help="global = pary OpenCV + globalne wyrownanie (domyslna, "
                            "odporna na obrot kadrow); detail = pipeline cv2.detail; "
                            "basic = cv2.Stitcher")
        p.add_argument("--okno", type=int, default=0,
                       help="[detail] 0 = wszystkie zdjecia (pelna mozaika), "
                            "N = tylko ostatnie N (staly czas, podglad na zywo)")
        p.add_argument("--megapix", type=float, default=0.6,
                       help="[detail] rozdzielczosc robocza do dopasowania")
        p.add_argument("--conf", type=float, default=1.0,
                       help="[detail] prog pewnosci powiazania zdjec; 1.0 jak w "
                            "cv2.Stitcher SCANS - nizszy przepuszcza bledne "
                            "polaczenia przy powtarzalnych rzedach uprawy")
        p.add_argument("--detektor", default="ORB", choices=["ORB", "SIFT"])
        p.add_argument("--model", default="affine",
                       choices=["affine", "rotacyjny"],
                       help="affine wlasciwy dla przelotu nad plaskim terenem; "
                            "rotacyjny tylko dla klasycznych panoram")
        p.add_argument("--zakres-par", type=int, default=0,
                       help="[detail] dopasowuj tylko zdjecia odlegle o <= N w kolejnosci; "
                            "0 = kazde z kazdym (wolne przy wielu zdjeciach)")
        p.add_argument("--renderuj-co", type=int, default=1,
                       help="[detail] renderuj mozaike co N zdjec")
        p.add_argument("--downscale", type=float, default=0.5,
                       help="[basic] przeskalowanie zdjec")
        p.add_argument("--tryb", default="scans", choices=["scans", "panorama"],
                       help="[basic] scans dla przelotu nad terenem")
        p.add_argument("--bez-ekspozycji", action="store_true",
                       help="[detail] wylacz wyrownanie jasnosci")
        p.add_argument("--cechy", type=int, default=1500,
                       help="[global] liczba punktow charakterystycznych na kadr")
        p.add_argument("--bloki-ekspozycji", action="store_true",
                       help="[global] blokowe wyrownanie jasnosci - dokladniejsze lokalnie, "
                            "ale wielokrotnie wolniejsze")
        p.add_argument("--szwy", action="store_true",
                       help="[detail] wlacz wyznaczanie szwow (ladniej, ale wolno)")
        p.add_argument("--kroki", action="store_true",
                       help="zapisuj mozaike po kazdym zdjeciu")

    p = sub.add_parser("batch", help="przetworz caly katalog naraz")
    wspolne(p)
    p.add_argument("--limit", type=int, help="ogranicz liczbe zdjec")
    p.add_argument("--opoznienie", type=float, default=0.0,
                   help="odstep miedzy zdjeciami w sekundach (symulacja lotu)")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("watch", help="obserwuj katalog i skladaj na biezaco")
    wspolne(p)
    p.add_argument("--interwal", type=float, default=1.0,
                   help="co ile sekund sprawdzac katalog")
    p.add_argument("--od-poczatku", action="store_true",
                   help="uwzglednij pliki obecne przy starcie")
    p.add_argument("--timeout", type=float, default=None,
                   help="zakoncz po N sekundach bez nowych zdjec")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("info", help="informacje o zbiorze")
    p.add_argument("katalog")
    p.set_defaults(func=cmd_info)

    # opcje symulatora lotu - wspolne dla rozloz i demo
    def ciecie(p):
        p.add_argument("obraz", nargs="?", help="duze zdjecie zrodlowe; brak = sztuczne pole")
        p.add_argument("--max-wymiar", type=int, default=0,
                       help="zmniejsz zrodlo, jesli dluzszy bok jest wiekszy")
        p.add_argument("--pole-w", type=int, default=2400, help="szerokosc sztucznego pola")
        p.add_argument("--pole-h", type=int, default=1600, help="wysokosc sztucznego pola")
        p.add_argument("--tryb-ciecia", default="trasa", choices=["trasa", "losowy"],
                       help="trasa = przelot tam i z powrotem, losowy = losowe miejsca")
        p.add_argument("--kadr", type=int, default=640, help="rozmiar kadru w px")
        p.add_argument("--zasieg", type=float, default=0,
                       help="ile px zrodla obejmuje kadr (0 = tyle co --kadr)")
        p.add_argument("--overlap", type=float, default=0.65, help="pokrycie kolejnych kadrow")
        p.add_argument("--sidelap", type=float, default=0.5, help="pokrycie miedzy liniami")
        p.add_argument("--liczba", type=int, default=0, help="[losowy] liczba kadrow")
        p.add_argument("--jitter", type=float, default=0.08, help="losowe odchylenie pozycji")
        p.add_argument("--obrot", type=float, default=0.0, help="losowe odchylenie kursu, stopnie")
        p.add_argument("--obrot-linii", action="store_true",
                       help="co druga linia obrocona o 180 st.")
        p.add_argument("--skala", type=float, default=0.0, help="losowe zmiany wysokosci lotu")
        p.add_argument("--ekspozycja", type=float, default=0.04, help="zmiennosc jasnosci")
        p.add_argument("--rozmycie", type=float, default=0.0,
                       help="prawdopodobienstwo rozmycia ruchowego")
        p.add_argument("--szum", type=float, default=0.0, help="szum matrycy")
        p.add_argument("--winieta", type=float, default=0.0,
                       help="przyciemnienie rogow kadru 0-1, typowe dla kamer dronowych")
        p.add_argument("--seed", type=int, default=7)

    p = sub.add_parser("rozloz", help="potnij jedno zdjecie na kadry (symulacja lotu)")
    ciecie(p)
    p.add_argument("-o", "--output", default="symulacja")
    p.add_argument("--na-zywo", help="wypuszczaj kadry do tego katalogu jak w locie")
    p.add_argument("--odstep", type=float, default=1.0, help="[na-zywo] sekundy miedzy kadrami")
    p.set_defaults(func=cmd_rozloz)

    p = sub.add_parser("ocen", help="porownaj mozaike z oryginalem")
    p.add_argument("mozaika")
    p.add_argument("oryginal")
    p.add_argument("--prawda", help="prawda.json z komendy rozloz")
    p.add_argument("-o", "--output", default="ocena")
    p.set_defaults(func=cmd_ocen)

    p = sub.add_parser("demo", help="rozloz -> zloz -> ocen jednym poleceniem")
    wspolne_bez_katalogu(p)
    ciecie(p)
    p.add_argument("-o", "--output", default="demo")
    p.set_defaults(func=cmd_demo)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s")
    try:
        return args.func(args)
    except Exception as e:
        log.error("%s", e)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
