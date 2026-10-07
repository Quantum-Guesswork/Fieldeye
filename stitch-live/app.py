"""GUI do symulatora lotu i skladania zdjec.

Uruchomienie (z katalogu projektu): streamlit run app.py


Zakladki:
  1. Rozklad zdjecia  - ciecie zdjecia
  2. Kadry            - gotowe kadry
  3. Skladanie        - symulacja laczenia kadr po kadrze, pobieranie wyniku
"""

import io
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from stitchlive.rozklad import Ustawienia, rozloz
from stitchlive.pole import zbuduj_pole
from stitchlive.ocena import ocen
from stitchlive.globalny import SkladaczGlobalny
from stitchlive.detail import SkladaczDetail
from stitchlive.basic import SkladaczPodstawowy


st.set_page_config(page_title="FIELDEYE - symulator lotu", layout="wide")

ROZSZERZENIA = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# gotowe zestawy zaklocen - zeby nie ustawiac suwakow recznie
PRESETY = {
    "Czyste ciecie": dict(obrot=0.0, skala=0.0, ekspozycja=0.0, rozmycie=0.0,
                          szum=0.0, winieta=0.0, jitter=0.0),
    "Realistyczny dron": dict(obrot=3.0, skala=0.03, ekspozycja=0.05, rozmycie=0.1,
                              szum=2.0, winieta=0.3, jitter=0.1),
    # na granicy mozliwosci metody - przy mocniejszych zakloceniach kadry
    # zaczynaja odpadac z mozaiki (widac to w metryce "W mozaice")
    "Trudne warunki": dict(obrot=6.0, skala=0.05, ekspozycja=0.07, rozmycie=0.25,
                           szum=4.0, winieta=0.35, jitter=0.12),
}




# Kazda karta przegladarki dostaje wlasny katalog na pliki - dwie osoby
# korzystajace z aplikacji naraz nie nadpisza sobie kadrow.
def katalog_roboczy():
    """Osobny katalog tymczasowy na sesje przegladarki."""
    if "workdir" not in st.session_state:
        st.session_state.workdir = Path(tempfile.mkdtemp(prefix="fieldeye_"))
    return st.session_state.workdir


# Wywolywane przez selectbox (on_change) - wpisuje wartosci zestawu do
# session_state, a suwaki z tymi samymi kluczami (key=...) je pokazuja.
def ustaw_preset():
    for k, v in PRESETY[st.session_state.preset].items():
        st.session_state[k] = v


# setdefault - ustawia wartosc tylko przy pierwszym uruchomieniu, nie nadpisuje zmian
def inicjuj_suwaki():
    for k, v in PRESETY["Realistyczny dron"].items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("preset", "Realistyczny dron")


# ---------------- pomocnicze ----------------

# OpenCV trzyma kolory jako BGR, przegladarka oczekuje RGB - bez tego niebo byloby pomaranczowe
def bgr_na_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# plik z uploadu to bajty w pamieci - imdecode dekoduje je bez zapisu na dysk
def wczytaj_z_uploadu(plik):
    dane = np.frombuffer(plik.getvalue(), np.uint8)
    return cv2.imdecode(dane, cv2.IMREAD_COLOR)


def zmniejsz(img, max_wymiar):
    if max_wymiar and max(img.shape[:2]) > max_wymiar:
        k = max_wymiar / max(img.shape[:2])
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return img


# ZIP budowany w pamieci (BytesIO) - od razu do przycisku pobierania,
# bez zostawiania plikow na dysku serwera.
def spakuj(pliki, dodatkowe=None):
    """ZIP w pamieci. pliki: lista sciezek, dodatkowe: {nazwa_w_zipie: bajty}."""
    bufor = io.BytesIO()
    with zipfile.ZipFile(bufor, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pliki:
            z.write(p, arcname=f"images/{Path(p).name}")
        for nazwa, dane in (dodatkowe or {}).items():
            z.writestr(nazwa, dane)
    return bufor.getvalue()


def jpg_bajty(img, jakosc=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jakosc])
    return buf.tobytes()


def gif_z_klatek(klatki, szer=700, ms=500):
    """Animacja rosnacej mozaiki. Wszystkie klatki na wspolnym plotnie,
    zeby animacja nie skakala przy zmianie rozmiaru mozaiki."""
    if not klatki:
        return None
    H = max(k.shape[0] for k in klatki)
    W = max(k.shape[1] for k in klatki)
    s = szer / W
    obrazy = []
    for k in klatki:
        plotno = np.zeros((H, W, 3), np.uint8)
        plotno[:k.shape[0], :k.shape[1]] = k
        plotno = cv2.resize(plotno, (szer, max(1, int(H * s))), interpolation=cv2.INTER_AREA)
        obrazy.append(Image.fromarray(bgr_na_rgb(plotno)))
    bufor = io.BytesIO()
    obrazy[0].save(bufor, format="GIF", save_all=True, append_images=obrazy[1:],
                   duration=ms, loop=0)
    return bufor.getvalue()


# Pomijamy __MACOSX (smieci, ktore macOS dokleja do ZIP-ow) i pliki,
# ktore nie sa obrazami.
def rozpakuj_kadry(pliki_uploadu, cel):
    """Kadry wgrane przez uzytkownika: pojedyncze obrazy albo ZIP."""
    if cel.exists():
        shutil.rmtree(cel)
    cel.mkdir(parents=True)
    for f in pliki_uploadu:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                for n in z.namelist():
                    if Path(n).suffix.lower() in ROZSZERZENIA and not n.startswith("__MACOSX"):
                        (cel / Path(n).name).write_bytes(z.read(n))
        elif Path(f.name).suffix.lower() in ROZSZERZENIA:
            (cel / f.name).write_bytes(f.getvalue())
    return sorted(p for p in cel.iterdir() if p.suffix.lower() in ROZSZERZENIA)


# =====================================================================
inicjuj_suwaki()
st.title("FIELDEYE — symulator lotu i składanie zdjęć")
st.caption("Tniesz jedno zdjęcie na kadry tak, jakby robił je dron, pobierasz je, "
           "a potem sprawdzasz, czy program złoży je z powrotem.")

# trzy zakladki - kod kazdej jest w bloku 'with tabN:' ponizej
tab1, tab2, tab3 = st.tabs(["1. Rozkład zdjęcia", "2. Kadry do pobrania", "3. Składanie"])

# =====================================================================
# 1. ROZKLAD
# =====================================================================
# =====================================================================
# ZAKLADKA 1: parametry po lewej, wynik po prawej
# =====================================================================
with tab1:
    lewa, prawa = st.columns([1, 2])

    with lewa:
        st.subheader("Źródło")
        zrodlo_typ = st.radio("Obraz do pocięcia", ["Wgraj zdjęcie", "Sztuczne pole"],
                              horizontal=True)
        if zrodlo_typ == "Wgraj zdjęcie":
            plik = st.file_uploader("Duże zdjęcie (ortofotomapa, zdjęcie z drona...)",
                                    type=["jpg", "jpeg", "png", "tif", "tiff"])
        else:
            c1, c2 = st.columns(2)
            pole_w = c1.number_input("Szerokość pola", 800, 6000, 2400, 100)
            pole_h = c2.number_input("Wysokość pola", 600, 6000, 1600, 100)
        max_wym = st.slider("Zmniejsz źródło do (dłuższy bok, px)", 800, 6000, 2400, 100,
                            help="Duże zdjęcia z drona mają 4000+ px - zmniejszenie "
                                 "przyspiesza wszystko kilkukrotnie")

        st.subheader("Lot")
        tryb = st.radio("Tryb cięcia", ["trasa", "losowy"], horizontal=True,
                        help="trasa = przelot tam i z powrotem, losowy = losowe miejsca "
                             "z zachowanym pokryciem między kolejnymi kadrami")
        kadr = st.select_slider("Rozmiar kadru (px)", [320, 400, 480, 560, 640, 800], 480)
        overlap = st.slider("Pokrycie kolejnych kadrów", 0.3, 0.9, 0.65, 0.05)
        if tryb == "trasa":
            sidelap = st.slider("Pokrycie między liniami", 0.2, 0.9, 0.5, 0.05)
            obrot_linii = st.checkbox("Co druga linia obrócona o 180° (dron zawraca)")
            liczba = 0
        else:
            sidelap, obrot_linii = 0.5, False
            liczba = st.slider("Liczba kadrów", 5, 150, 30)

        st.subheader("Wygląd jak z drona")
        st.selectbox("Zestaw", list(PRESETY), key="preset", on_change=ustaw_preset)
        with st.expander("Szczegóły zakłóceń", expanded=False):
            st.slider("Odchylenie kursu (°)", 0.0, 20.0, key="obrot", step=0.5)
            st.slider("Zmiana wysokości lotu", 0.0, 0.15, key="skala", step=0.01)
            st.slider("Zmienność jasności", 0.0, 0.2, key="ekspozycja", step=0.01)
            st.slider("Szansa rozmycia ruchowego", 0.0, 1.0, key="rozmycie", step=0.05)
            st.slider("Szum matrycy", 0.0, 15.0, key="szum", step=0.5)
            st.slider("Winietowanie (ciemne rogi)", 0.0, 0.8, key="winieta", step=0.05)
            st.slider("Losowe odchylenie pozycji", 0.0, 0.4, key="jitter", step=0.01)
        seed = st.number_input("Ziarno losowania", 0, 99999, 7,
                               help="To samo ziarno = te same kadry")

        uruchom = st.button("Rozłóż zdjęcie", type="primary", width="stretch")

    with prawa:
        # Ten blok wykonuje sie tylko w przebiegu, w ktorym kliknieto przycisk.
        # Wynik trafia do session_state, wiec zostaje widoczny po kolejnych kliknieciach.
        if uruchom:
            if zrodlo_typ == "Wgraj zdjęcie":
                if plik is None:
                    st.warning("Najpierw wgraj zdjęcie.")
                    st.stop()
                img = wczytaj_z_uploadu(plik)
                if img is None:
                    st.error("Nie udało się odczytać obrazu.")
                    st.stop()
            else:
                img = zbuduj_pole(int(pole_w), int(pole_h), seed=int(seed))
            img = zmniejsz(img, max_wym)

            u = Ustawienia(tryb=tryb, kadr=int(kadr), overlap=overlap, sidelap=sidelap,
                           liczba=int(liczba), jitter=st.session_state.jitter,
                           obrot=st.session_state.obrot, obrot_linii=obrot_linii,
                           skala=st.session_state.skala,
                           ekspozycja=st.session_state.ekspozycja,
                           rozmycie=st.session_state.rozmycie, szum=st.session_state.szum,
                           winieta=st.session_state.winieta, seed=int(seed))
            out = katalog_roboczy() / "symulacja"
            try:
                with st.spinner("Tnę zdjęcie na kadry..."):
                    prawda = rozloz(img, out, u)
            except ValueError as e:
                st.error(f"{e}. Zmniejsz rozmiar kadru albo zwiększ źródło.")
                st.stop()
            # Zapamietujemy: gdzie sa kadry, prawde (do oceny) i oryginal (do porownania).
            st.session_state.symulacja = {"katalog": out, "prawda": prawda, "oryginal": img}
            st.session_state.pop("wynik", None)     # stary wynik składania nieaktualny

        sym = st.session_state.get("symulacja")
        if sym:
            k = sym["prawda"]["kadry"]
            st.success(f"Gotowe: {len(k)} kadrów {sym['prawda']['ustawienia']['kadr']} px. "
                       "Przejdź do zakładki 2, żeby je pobrać, albo 3, żeby je złożyć.")
            st.image(bgr_na_rgb(cv2.imread(str(sym["katalog"] / "podglad_trasy.jpg"))),
                     caption="Czerwone ramki = kadry, żółta linia = kolejność lotu",
                     width="stretch")
            przyklad = [sym["katalog"] / "images" / x["plik"] for x in k[:4]]
            st.caption("Pierwsze kadry:")
            kol = st.columns(len(przyklad))
            for c, p in zip(kol, przyklad):
                c.image(bgr_na_rgb(cv2.imread(str(p))), caption=p.name,
                        width="stretch")
        else:
            st.info("Ustaw parametry po lewej i kliknij **Rozłóż zdjęcie**.")

# =====================================================================
# 2. KADRY
# =====================================================================
# =====================================================================
# ZAKLADKA 2: przegladanie i pobieranie kadrow z zakladki 1
# =====================================================================
with tab2:
    sym = st.session_state.get("symulacja")
    if not sym:
        st.info("Najpierw rozłóż zdjęcie w zakładce 1.")
    else:
        kat = sym["katalog"]
        pliki = sorted((kat / "images").glob("*.jpg"))
        rozmiar_mb = sum(p.stat().st_size for p in pliki) / 1e6

        c1, c2, c3 = st.columns(3)
        c1.metric("Kadrów", len(pliki))
        c2.metric("Rozmiar kadru", f"{sym['prawda']['ustawienia']['kadr']} px")
        c3.metric("Łącznie", f"{rozmiar_mb:.1f} MB")

        dodatki = {
            "prawda.json": json.dumps(sym["prawda"], indent=1),
            "podglad_trasy.jpg": (kat / "podglad_trasy.jpg").read_bytes(),
            "oryginal.jpg": (kat / "oryginal.jpg").read_bytes(),
        }
        d1, d2 = st.columns(2)
        d1.download_button("Pobierz wszystkie kadry (ZIP)", spakuj(pliki),
                           file_name="kadry.zip", mime="application/zip",
                           type="primary", width="stretch")
        d2.download_button("Pobierz kadry + prawda.json + podgląd trasy (ZIP)",
                           spakuj(pliki, dodatki), file_name="symulacja_pelna.zip",
                           mime="application/zip", width="stretch")
        st.caption("`prawda.json` zawiera prawdziwe położenie, kąt i zasięg każdego kadru "
                   "w oryginale - przydaje się do sprawdzenia dokładności składania.")

        st.divider()
        # galeria stronicowana - setka miniatur naraz spowolnilaby przegladarke
        na_strone = 24
        stron = max(1, (len(pliki) + na_strone - 1) // na_strone)
        strona = st.number_input("Strona galerii", 1, stron, 1) if stron > 1 else 1
        widoczne = pliki[(strona - 1) * na_strone: strona * na_strone]
        kolumny = st.columns(6)
        for i, p in enumerate(widoczne):
            kolumny[i % 6].image(bgr_na_rgb(cv2.imread(str(p))), caption=p.name,
                                 width="stretch")

        st.divider()
        wybrany = st.selectbox("Podgląd pojedynczego kadru", [p.name for p in pliki])
        p = kat / "images" / wybrany
        info = next(x for x in sym["prawda"]["kadry"] if x["plik"] == wybrany)
        a, b = st.columns([2, 1])
        a.image(bgr_na_rgb(cv2.imread(str(p))), width="stretch")
        b.json({"srodek_w_oryginale_px": info["srodek"], "kat_st": info["kat_st"],
                "zasieg_px": info["zasieg_px"]})
        b.download_button(f"Pobierz {wybrany}", p.read_bytes(), file_name=wybrany,
                          mime="image/jpeg", width="stretch")

# =====================================================================
# 3. SKLADANIE
# =====================================================================
# =====================================================================
# ZAKLADKA 3: skladanie kadrow w mozaike z podgladem na zywo
# =====================================================================
with tab3:
    lewa, prawa = st.columns([1, 2])

    with lewa:
        st.subheader("Kadry do złożenia")
        opcje = ["Z zakładki 1 (symulacja)", "Wgraj własne zdjęcia / ZIP"]
        zrodlo = st.radio("Źródło", opcje,
                          index=0 if st.session_state.get("symulacja") else 1)
        wgrane = None
        if zrodlo == opcje[1]:
            wgrane = st.file_uploader("Zdjęcia w kolejności lotu albo jeden ZIP",
                                      type=["jpg", "jpeg", "png", "tif", "tiff", "zip"],
                                      accept_multiple_files=True)

        st.subheader("Metoda")
        metoda = st.radio("Algorytm", ["global", "detail", "basic"], horizontal=True,
                          help="global - odporna na obrót kadrów (domyślna); "
                               "detail - pipeline cv2.detail; basic - cv2.Stitcher")
        megapix = st.slider("Rozdzielczość robocza (Mpx)", 0.1, 2.0, 0.6, 0.1)
        renderuj_co = st.slider("Odświeżaj mozaikę co N kadrów", 1, 20, 3,
                                help="Mniejsze N = płynniejszy podgląd, ale wolniej")
        zakres = st.number_input("Dopasowuj z N poprzednimi kadrami (0 = z każdym)", 0, 200, 0)
        szwy = st.checkbox("Wyznaczaj szwy (ładniej, wolniej)")
        opoznienie = st.slider("Odstęp między kadrami (s) - symulacja lotu", 0.0, 2.0, 0.0, 0.1)
        zloz = st.button("Złóż mozaikę", type="primary", width="stretch")

    with prawa:
        if zloz:
            if zrodlo == opcje[0]:
                sym = st.session_state.get("symulacja")
                if not sym:
                    st.warning("Brak symulacji - rozłóż zdjęcie w zakładce 1.")
                    st.stop()
                sciezki = sorted((sym["katalog"] / "images").glob("*.jpg"))
            else:
                if not wgrane:
                    st.warning("Wgraj zdjęcia albo ZIP.")
                    st.stop()
                sciezki = rozpakuj_kadry(wgrane, katalog_roboczy() / "wgrane")
                sym = None
            if len(sciezki) < 2:
                st.warning("Potrzebne są co najmniej dwa zdjęcia.")
                st.stop()

            if metoda == "global":
                # wszystkie trzy skladacze maja te sama metode dodaj() - reszta kodu jest wspolna
                s = SkladaczGlobalny(work_megapix=megapix, zakres_par=int(zakres),
                                     szwy=szwy, renderuj_co=renderuj_co)
            elif metoda == "detail":
                s = SkladaczDetail(work_megapix=megapix, zakres_par=int(zakres),
                                   szwy=szwy, renderuj_co=renderuj_co)
            else:
                s = SkladaczPodstawowy(downscale=min(1.0, megapix / 0.6 * 0.5))

            pasek = st.progress(0.0, text="Start...")
            podglad = st.empty()
            # Petla kadr po kadrze. st.empty() to 'miejsce' na stronie, ktore nadpisujemy
            # nowym obrazem - dlatego mozaika rosnie w tym samym miejscu zamiast dopisywac
            # kolejne obrazki pod spodem. klatki - kolejne stany mozaiki do animacji GIF.
            klatki, t0 = [], time.time()
            for i, p in enumerate(sciezki, start=1):
                ok = s.dodaj(p)
                if ok and s.mozaika is not None:
                    klatki.append(s.mozaika.copy())
                    podglad.image(bgr_na_rgb(s.mozaika), width="stretch",
                                  caption=f"Mozaika po {i} z {len(sciezki)} kadrów")
                pasek.progress(i / len(sciezki), text=f"Kadr {i}/{len(sciezki)}: {p.name}")
                if opoznienie:
                    time.sleep(opoznienie)
            if hasattr(s, "zakoncz") and s.zakoncz() and s.mozaika is not None:
                if not klatki or klatki[-1].shape != s.mozaika.shape or \
                        not np.array_equal(klatki[-1], s.mozaika):
                    klatki.append(s.mozaika.copy())
            czas = time.time() - t0
            pasek.empty()
            podglad.empty()

            if s.mozaika is None:
                st.error("Nie udało się złożyć mozaiki. Zwiększ pokrycie kadrów albo "
                         "spróbuj metody global.")
                st.stop()

            wynik = {"mozaika": s.mozaika, "klatki": klatki, "czas": czas,
                     "n": len(sciezki), "metoda": metoda,
                     "raport": s.raport() if hasattr(s, "raport") else None}
            if sym:
                wynik["ocena"] = ocen(s.mozaika, sym["oryginal"], sym["prawda"],
                                      outdir=katalog_roboczy() / "ocena")
            st.session_state.wynik = wynik

        w = st.session_state.get("wynik")
        if w:
            h, wd = w["mozaika"].shape[:2]
            c = st.columns(4)
            c[0].metric("Kadrów", w["n"])
            c[1].metric("Czas", f"{w['czas']:.1f} s")
            c[2].metric("Mozaika", f"{wd}×{h}")
            if w.get("raport"):
                c[3].metric("W mozaice", f"{w['raport']['kadrow_w_mozaice']}/{w['n']}")

            st.image(bgr_na_rgb(w["mozaika"]), caption=f"Mozaika ({w['metoda']})",
                     width="stretch")

            o = w.get("ocena")
            if o and o.get("ok"):
                st.subheader("Porównanie z oryginałem")
                m = st.columns(4)
                m[0].metric("PSNR", f"{o['psnr_db']} dB", help="powyżej ~25 dB = bardzo dobrze")
                m[1].metric("Pokrycie", f"{o['pokrycie']:.1%}")
                m[2].metric("Skala", o["skala"], help="1.0 = zgodna z oryginałem")
                m[3].metric("Błąd średni", o["blad_sredni"])
                st.write(f"**Werdykt:** {o['werdykt']}")
                nal = katalog_roboczy() / "ocena" / "ocena_nalozenie.jpg"
                if nal.exists():
                    st.image(bgr_na_rgb(cv2.imread(str(nal))),
                             caption="Mozaika nałożona na oryginał", width="stretch")
            elif o:
                st.warning(f"Ocena nieudana: {o.get('powod')}")

            st.subheader("Pobierz")
            gif = gif_z_klatek(w["klatki"])
            raport = {"metoda": w["metoda"], "kadrow": w["n"], "czas_s": round(w["czas"], 2),
                      "rozmiar_px": [wd, h], "wyrownanie": w.get("raport"),
                      "ocena": {k: v for k, v in (o or {}).items() if k != "macierz"}}
            d = st.columns(3)
            d[0].download_button("Mozaika (JPG)", jpg_bajty(w["mozaika"]),
                                 file_name="mozaika.jpg", mime="image/jpeg",
                                 type="primary", width="stretch")
            if gif:
                d[1].download_button("Animacja składania (GIF)", gif,
                                     file_name="skladanie.gif", mime="image/gif",
                                     width="stretch")
            # ZIP: kazdy kolejny stan mozaiki jako osobny JPG + koncowa mozaika + raport
            kroki = {f"kroki/krok_{i:03d}.jpg": jpg_bajty(k, 85)
                     for i, k in enumerate(w["klatki"], start=1)}
            kroki["mozaika.jpg"] = jpg_bajty(w["mozaika"])
            kroki["raport.json"] = json.dumps(raport, indent=2, ensure_ascii=False,
                                              default=str)
            d[2].download_button("Wszystko: mozaika + kroki + raport (ZIP)",
                                 spakuj([], kroki), file_name="skladanie.zip",
                                 mime="application/zip", width="stretch")
            if gif:
                with st.expander("Podgląd animacji składania"):
                    st.image(gif)
        else:
            st.info("Wybierz kadry i kliknij **Złóż mozaikę**. Podgląd będzie rosnął "
                    "kadr po kadrze, jak w trakcie lotu.")
