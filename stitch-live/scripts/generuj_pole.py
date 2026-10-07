"""Generator realistycznych zdjec pola z drona - proceduralnie, z etykietami.

W odroznieniu od obrazow z generatora AI tu wiadomo dokladnie, gdzie jest
kazdy chwast. Dostajesz od razu:
  - zdjecie pola (do testow skladania - mozna je potem pociac symulatorem),
  - etykiety chwastow w formacie YOLO (klasa cx cy w h, znormalizowane),
  - podglad z ramkami do sprawdzenia etykiet wzrokiem.

Uzycie:
    python scripts/generuj_pole.py dane_testowe --liczba 4
    python scripts/generuj_pole.py dane_testowe --uprawa kukurydza --szer 4000 --wys 2600
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

KLASY = {0: "chwast_szerokolistny", 1: "chwast_trawiasty"}

# Parametry kazdej uprawy. Kolory w BGR (kolejnosc OpenCV).
# pierzaste=True - liscie z bocznymi listkami (marchew), False - pelne liscie.
UPRAWY = {
    # rozstaw rzedow i roslin (px), promien kepki, liczba listkow, kolor bazowy (BGR)
    "marchew": dict(rzedy=175, rosliny=24, promien=50, listki=34, pierzaste=True,
                    kolor=(55, 150, 78), owoc=(30, 115, 225)),
    "kukurydza": dict(rzedy=230, rosliny=62, promien=58, listki=10, pierzaste=False,
                      kolor=(60, 140, 85), owoc=None),
    "burak": dict(rzedy=185, rosliny=44, promien=52, listki=12, pierzaste=False,
                  kolor=(55, 130, 65), owoc=None),
}


# ---------------- gleba ----------------

# Szum w kilku skalach naraz: losowa siatka jest powiekszana do rozmiaru
# obrazu (INTER_CUBIC wygladza), male skale = drobne ziarno, duze = plamy.
# Tak powstaje naturalna tekstura bez regularnego wzoru.
def szum_wieloskalowy(h, w, rng, skale=(4, 16, 64, 200), wagi=(0.15, 0.3, 0.35, 0.2)):
    """Suma rozmytego szumu w kilku skalach - daje naturalna, niejednolita teksture."""
    wynik = np.zeros((h, w), np.float32)
    for s, wg in zip(skale, wagi):
        mh, mw = max(2, h // s), max(2, w // s)
        maly = rng.normal(0, 1, (mh, mw)).astype(np.float32)
        wynik += wg * cv2.resize(maly, (w, h), interpolation=cv2.INTER_CUBIC)
    return wynik / (np.abs(wynik).max() + 1e-6)


def gleba(h, w, rng):
    # ciemnobrazowa gleba: w BGR najwiecej czerwieni, najmniej niebieskiego
    baza = np.array([38, 48, 60], np.float32)                 # ciemnobrazowa, BGR
    n = szum_wieloskalowy(h, w, rng)
    img = baza[None, None, :] * (1 + 0.28 * n[:, :, None])
    # grudki - drobne jasne i ciemne plamki
    for _ in range(h * w // 180):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        k = float(rng.choice([0.55, 0.7, 1.35, 1.5]))
        cv2.circle(img, (x, y), r, tuple(float(c) * k for c in baza), -1)
    # wilgoc - duze, lagodne ciemniejsze obszary
    wil = szum_wieloskalowy(h, w, rng, skale=(300,), wagi=(1.0,))
    img *= (1 - 0.12 * np.clip(wil, 0, 1))[:, :, None]
    return img


def slady_opon(img, rng, x_srodek, szer=90):
    """Pionowy pas ubitej ziemi z bieznikiem w jodelke, jak po traktorze."""
    h, w = img.shape[:2]
    x0, x1 = max(0, x_srodek - szer // 2), min(w, x_srodek + szer // 2)
    img[:, x0:x1] *= 1.08                                     # ubita ziemia jasniejsza
    krok = 22
    for kolo in (x0 + szer // 4, x1 - szer // 4):
        for y in range(-krok, h + krok, krok):
            yy = y + int(rng.integers(-2, 3))
            for strona in (-1, 1):
                p1 = (kolo, yy)
                p2 = (kolo + strona * szer // 5, yy - krok // 2)
                if rng.random() < 0.85:                      # miejscami zatarty
                    cv2.line(img, p1, p2, (30, 37, 46), 2, cv2.LINE_AA)


# ---------------- rosliny ----------------

SLONCE = (-0.6, -0.8)          # kierunek swiatla: z lewej gory


def kolor_zieleni(baza, rng, zmiennosc=28, ton=1.0):
    """Losowy odcien zieleni. ton < 1 = liscie w cieniu, > 1 = oswietlone."""
    b, g, r = baza
    d = rng.normal(0, zmiennosc)
    zolc = max(0.0, ton - 1.0) * 35                        # swiatlo zolci zielen
    return (float(np.clip((b + rng.normal(0, 8)) * ton, 0, 255)),
            float(np.clip((g + d) * ton, 30, 245)),
            float(np.clip((r + rng.normal(0, 10) + d * 0.3) * ton + zolc, 0, 255)))


# Rosliny rysujemy na osobnej warstwie z maska zamiast wprost na glebie.
# Dzieki temu mozna potem dodac cien pod roslinami i fakture tylko na lisciach.
class Warstwa:
    """Rysowanie roslin na osobnej warstwie z maska - potem nakladane na glebe."""

    def __init__(self, h, w):
        self.img = np.zeros((h, w, 3), np.float32)
        self.maska = np.zeros((h, w), np.float32)
        self.cien = np.zeros((h, w), np.float32)

    def elipsa(self, c, osie, kat, kolor):
        c = (int(c[0]), int(c[1]))
        osie = (max(1, int(osie[0])), max(1, int(osie[1])))
        cv2.ellipse(self.img, c, osie, kat, 0, 360, kolor, -1, cv2.LINE_AA)
        cv2.ellipse(self.maska, c, osie, kat, 0, 360, 1.0, -1, cv2.LINE_AA)

    def linia(self, p1, p2, kolor, grub):
        p1, p2 = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
        cv2.line(self.img, p1, p2, kolor, grub, cv2.LINE_AA)
        cv2.line(self.maska, p1, p2, 1.0, grub, cv2.LINE_AA)


def _lisc_pierzasty(W, cx, cy, kat, dl, R, baza, rng, ton, grub_listka):
    ex, ey = cx + dl * math.cos(kat), cy + dl * math.sin(kat)
    W.linia((cx, cy), (ex, ey), kolor_zieleni(baza, rng, 15, ton * 0.85), 2)
    for t in np.linspace(0.25, 1.0, 6):
        px, py = cx + (ex - cx) * t, cy + (ey - cy) * t
        for s in (-1, 1):
            a2 = kat + s * rng.uniform(0.5, 1.0)
            l2 = R * 0.30 * (1.25 - t) * rng.uniform(0.8, 1.2)
            W.elipsa((px + l2 * math.cos(a2) / 2, py + l2 * math.sin(a2) / 2),
                     (l2 / 2, l2 * grub_listka), math.degrees(a2),
                     kolor_zieleni(baza, rng, 18, ton))


def kepka(W, cx, cy, u, rng, skala=1.0):
    """Roslina uprawna w trzech warstwach tonalnych: cien, srodek, swiatlo.

    Liscie od strony slonca sa jasniejsze i zolciej zielone, a spod kepki
    ciemny - to daje wrazenie objetosci, ktorego brakuje przy jednym kolorze.
    """
    R = u["promien"] * skala * rng.uniform(0.8, 1.2)
    cv2.circle(W.cien, (int(cx + 6), int(cy + 9)), int(R * 0.95), 1.0, -1)
    for warstwa, (ton, ile, zasieg) in enumerate([(0.62, 0.5, 1.0), (0.95, 0.5, 0.9), (1.25, 0.35, 0.7)]):
        for _ in range(max(3, int(u["listki"] * ile))):
            kat = rng.uniform(0, 2 * math.pi)
            dl = R * rng.uniform(0.45, 1.0) * zasieg
            # warstwa swiatla przesunieta w strone slonca
            sx = cx + (SLONCE[0] * R * 0.18 if warstwa == 2 else 0)
            sy = cy + (SLONCE[1] * R * 0.18 if warstwa == 2 else 0)
            if u["pierzaste"]:
                _lisc_pierzasty(W, sx, sy, kat, dl, R, u["kolor"], rng, ton, 0.2)
            else:
                mx, my = sx + dl / 2 * math.cos(kat), sy + dl / 2 * math.sin(kat)
                kol = kolor_zieleni(u["kolor"], rng, 20, ton)
                W.elipsa((mx, my), (dl / 2, R * 0.2), math.degrees(kat), kol)
                W.linia((sx, sy), (sx + dl * math.cos(kat), sy + dl * math.sin(kat)),
                        tuple(c * 0.75 for c in kol), 1)


def chwast_szerokolistny(W, cx, cy, rng):
    """Rozeta z kilku duzych zaokraglonych lisci - inny ksztalt i odcien niz uprawa."""
    R = rng.uniform(16, 30)
    cv2.circle(W.cien, (int(cx + 4), int(cy + 6)), int(R), 1.0, -1)
    n = int(rng.integers(4, 7))
    kat0 = rng.uniform(0, 2 * math.pi)
    baza = (40, float(rng.uniform(160, 190)), float(rng.uniform(95, 135)))
    for i in range(n):
        kat = kat0 + i * 2 * math.pi / n + rng.normal(0, 0.2)
        d = R * 0.55
        c = (cx + d * math.cos(kat), cy + d * math.sin(kat))
        # strona lisca od slonca jasniejsza
        ton = 1.0 + 0.25 * -(math.cos(kat) * SLONCE[0] + math.sin(kat) * SLONCE[1]) * -1
        W.elipsa(c, (R * 0.6, R * 0.38), math.degrees(kat), kolor_zieleni(baza, rng, 10, ton))
        W.linia((cx, cy), (cx + R * 0.95 * math.cos(kat), cy + R * 0.95 * math.sin(kat)),
                kolor_zieleni(baza, rng, 5, 0.75), 1)                 # nerw lisca
    W.elipsa((cx, cy), (R * 0.16, R * 0.16), 0, (35, 110, 65))
    return R * 1.15


def chwast_trawiasty(W, cx, cy, rng):
    """Kepa waskich, lekko zakrzywionych zdzbel - jak chwastnica."""
    R = rng.uniform(26, 44)
    cv2.circle(W.cien, (int(cx + 3), int(cy + 5)), int(R * 0.6), 0.6, -1)
    for _ in range(int(rng.integers(18, 30))):
        kat = rng.uniform(0, 2 * math.pi)
        dl = R * rng.uniform(0.6, 1.0)
        kol = kolor_zieleni((70, 165, 125), rng, 15, rng.uniform(0.8, 1.2))
        mx, my = cx + dl * 0.5 * math.cos(kat), cy + dl * 0.5 * math.sin(kat)
        kat2 = kat + rng.normal(0, 0.35)
        ex, ey = mx + dl * 0.5 * math.cos(kat2), my + dl * 0.5 * math.sin(kat2)
        W.linia((cx, cy), (mx, my), kol, 3)
        W.linia((mx, my), (ex, ey), kol, 2)
    return R


# ---------------- skladanie sceny ----------------

# Sklada cala scene: gleba -> slady opon -> rzedy uprawy -> linia kroplujaca
# -> chwasty (z zapisem ramek) -> cien -> nalozenie roslin -> oswietlenie.
def generuj(szer, wys, uprawa="marchew", gestosc_chwastow=1.0, seed=0):
    rng = np.random.default_rng(seed)
    u = UPRAWY[uprawa]

    img = gleba(wys, szer, rng)
    W = Warstwa(wys, szer)

    # slady opon co kilkaset pikseli, poprzecznie do rzedow
    for x in range(int(rng.integers(500, 900)), szer, int(rng.integers(700, 1000))):
        slady_opon(img, rng, x)

    # rzedy uprawy z niewielkim falowaniem - siewnik nie jedzie idealnie prosto
    # polozenia rzedow z lekko losowym rozstawem
    rzedy_y = []
    y = u["rzedy"] // 2 + int(rng.integers(0, u["rzedy"] // 3))
    while y < wys + u["rzedy"]:
        rzedy_y.append(y)
        y += u["rzedy"] + int(rng.normal(0, 4))

    faza = rng.uniform(0, 2 * math.pi)
    for ry in rzedy_y:
        x = int(rng.integers(0, u["rosliny"]))
        while x < szer + u["rosliny"]:
            fy = ry + 8 * math.sin(x / 400 + faza) + rng.normal(0, 3)
            if rng.random() > 0.04:                                 # pojedyncze wypady
                kepka(W, x + rng.normal(0, 3), fy, u, rng)
                if u["owoc"] and rng.random() < 0.06:               # wystajaca marchew
                    ox = int(x + rng.normal(0, 10))
                    oy = int(fy + rng.choice([-1, 1]) * u["promien"] * 0.9)
                    W.elipsa((ox, oy), (8, 3), rng.uniform(0, 180),
                             tuple(c * rng.uniform(0.85, 1.1) for c in u["owoc"]))
            x += u["rosliny"] + int(rng.normal(0, 3))

    # linia kroplujaca przy co ktoryms rzedzie
    for ry in rzedy_y[1::3]:
        pts = np.array([[x, ry + u["rzedy"] * 0.42 + 3 * math.sin(x / 250)]
                        for x in range(0, szer + 20, 20)], np.int32)
        cv2.polylines(img, [pts], False, (60, 40, 25), 3, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, (95, 70, 45), 1, cv2.LINE_AA)

    # chwasty - glownie w miedzyrzedziach, czesc w samym rzedzie (trudniejsze)
    etykiety = []
    # liczba chwastow proporcjonalna do powierzchni
    n_chw = int(szer * wys / 90_000 * gestosc_chwastow)
    for _ in range(n_chw):
        cx = rng.uniform(20, szer - 20)
        if rng.random() < 0.75:
            ry = rzedy_y[int(rng.integers(0, len(rzedy_y)))]
            cy = ry + u["rzedy"] * rng.uniform(0.3, 0.7)            # miedzyrzedzie
        else:
            cy = rzedy_y[int(rng.integers(0, len(rzedy_y)))] + rng.normal(0, 10)
        if cy < 20 or cy > wys - 20:
            continue
        if rng.random() < 0.6:
            R, klasa = chwast_szerokolistny(W, cx, cy, rng), 0
        else:
            R, klasa = chwast_trawiasty(W, cx, cy, rng), 1
        x0, y0 = max(0, cx - R), max(0, cy - R)
        x1, y1 = min(szer, cx + R), min(wys, cy + R)
        # Ramka chwastu: kwadrat o boku 2R wokol srodka, przyciety do obrazu.
        # To jest 'prawda' - dokladnie tam narysowalismy chwast.
        etykiety.append({"klasa": klasa, "nazwa": KLASY[klasa],
                         "xyxy": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]})

    # cien pod roslinami - rzucany na glebe, rozmyty
    cien = cv2.GaussianBlur(W.cien, (0, 0), 7)
    img *= (1 - 0.45 * np.clip(cien, 0, 1))[:, :, None]

    # faktura lisci: drobny szum tylko na roslinach
    faktura = 1 + 0.12 * szum_wieloskalowy(wys, szer, rng, skale=(2, 5), wagi=(0.6, 0.4))
    rosl = W.img * faktura[:, :, None]
    # ciemniejsze szczeliny miedzy liscmi - tam, gdzie maska jest slaba
    m = np.clip(W.maska, 0, 1)
    m_miekka = cv2.GaussianBlur(m, (0, 0), 0.8)
    # Nalozenie roslin na glebe: gdzie maska = 1 - roslina, gdzie 0 - gleba,
    # pomiedzy (krawedzie lisci) - plynne przejscie.
    img = img * (1 - m_miekka[:, :, None]) + rosl * m_miekka[:, :, None]

    # oswietlenie: lagodny gradient + lekki spadek ku krawedziom (obiektyw)
    yy, xx = np.mgrid[0:wys, 0:szer].astype(np.float32)
    grad = 1 + 0.06 * ((xx / szer) - 0.5) + 0.04 * ((yy / wys) - 0.5)
    img *= grad[:, :, None]

    img = cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), 0.6)      # optyka nie jest idealna
    img = np.clip(img + rng.normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
    return img, etykiety


def zapisz(img, etykiety, katalog, nazwa):
    katalog = Path(katalog)
    for sub in ("images", "labels", "podglad"):
        (katalog / sub).mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]

    cv2.imwrite(str(katalog / "images" / f"{nazwa}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])

    # YOLO: klasa cx cy w h - znormalizowane do 0..1
    # format YOLO: jedna linia na obiekt, wartosci jako ulamek szerokosci/wysokosci obrazu
    wiersze = []
    for e in etykiety:
        x0, y0, x1, y1 = e["xyxy"]
        wiersze.append(f"{e['klasa']} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                       f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    (katalog / "labels" / f"{nazwa}.txt").write_text("\n".join(wiersze) + "\n")

    pod = img.copy()
    for e in etykiety:
        x0, y0, x1, y1 = map(int, e["xyxy"])
        kol = (0, 0, 255) if e["klasa"] == 0 else (255, 0, 255)
        cv2.rectangle(pod, (x0, y0), (x1, y1), kol, 2)
    cv2.imwrite(str(katalog / "podglad" / f"{nazwa}.jpg"), pod, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("katalog")
    ap.add_argument("--liczba", type=int, default=3, help="ile zdjec wygenerowac")
    ap.add_argument("--szer", type=int, default=3600)
    ap.add_argument("--wys", type=int, default=2200)
    ap.add_argument("--uprawa", default="marchew", choices=list(UPRAWY) + ["mix"])
    ap.add_argument("--chwasty", type=float, default=1.0, help="gestosc chwastow (1 = typowa)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    opis = []
    for i in range(a.liczba):
        uprawa = list(UPRAWY)[i % len(UPRAWY)] if a.uprawa == "mix" else a.uprawa
        nazwa = f"pole_{i + 1:02d}_{uprawa}"
        img, et = generuj(a.szer, a.wys, uprawa, a.chwasty, seed=a.seed + i)
        zapisz(img, et, a.katalog, nazwa)
        opis.append({"plik": f"images/{nazwa}.jpg", "uprawa": uprawa, "rozmiar": [a.szer, a.wys],
                     "chwasty": len(et), "szerokolistne": sum(e["klasa"] == 0 for e in et),
                     "trawiaste": sum(e["klasa"] == 1 for e in et)})
        print(f"{nazwa}: {len(et)} chwastow")

    Path(a.katalog).joinpath("classes.txt").write_text("\n".join(KLASY[k] for k in sorted(KLASY)) + "\n")
    Path(a.katalog).joinpath("opis.json").write_text(json.dumps(opis, indent=2, ensure_ascii=False))
    print(f"\nGotowe: {a.katalog}/images, etykiety YOLO w labels/, podglad z ramkami w podglad/")


if __name__ == "__main__":
    main()
