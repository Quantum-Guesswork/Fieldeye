"""Ocena mozaiki wzgledem oryginalu, z ktorego zostala rozlozona.

Mozaika ma wlasny, umowny uklad wspolrzednych - nie da sie jej porownac
z oryginalem piksel w piksel. Najpierw dopasowujemy ja do oryginalu
(cechy SIFT, awaryjnie ORB; RANSAC; przeksztalcenie podobienstwa), potem liczymy:

  zgodne_pary  - ile dopasowan potwierdza przeksztalcenie (wiarygodnosc)
  skala        - ~1.0 oznacza, ze mozaika ma skale oryginalu
  pokrycie     - jaka czesc obszaru przeleconego trafila do mozaiki
  blad_sredni  - srednia roznica jasnosci w czesci wspolnej (0-255)
  psnr_db      - to samo jako PSNR; powyzej ~25 dB to dobra zgodnosc
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


# Obraz w skali szarosci, zmniejszony do max_wymiar px. Zwraca tez wspolczynnik
# zmniejszenia k - potrzebny, zeby wspolrzedne punktow przeliczyc z powrotem.
def _szary(img, max_wymiar):
    k = min(1.0, max_wymiar / max(img.shape[:2]))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if k < 1.0:
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return g, k


def _pary_wzajemne(d1, d2, norm):
    """Test proporcji + dopasowanie wzajemne: para musi byc najlepsza
    w obie strony. Bez tego przy powtarzalnych rzedach wiele punktow
    mozaiki trafia w jeden punkt oryginalu i RANSAC wybiera model
    zdegenerowany (sciagniecie wszystkiego do jednego punktu)."""
    bf = cv2.BFMatcher(norm)
    # najlepsze dopasowanie w przeciwna strone: punkt oryginalu -> punkt mozaiki
    wstecz = {m.queryIdx: m.trainIdx for m in bf.match(d2, d1)}
    pary = []
    for para in bf.knnMatch(d1, d2, k=2):
        if len(para) == 2 and para[0].distance < 0.8 * para[1].distance:
            # para wzajemna: A wybralo B i B wybralo A
            if wstecz.get(para[0].trainIdx) == para[0].queryIdx:
                pary.append(para[0])
    return pary


def _obszar_lotu(prawda, W, H, zapas=0.05):
    """Prostokat obejmujacy wszystkie kadry, z niewielkim zapasem."""
    pts = np.array([p for k in prawda["kadry"] for p in k["naroza"]], float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    dx, dy = (x1 - x0) * zapas, (y1 - y0) * zapas
    return (int(max(0, x0 - dx)), int(max(0, y0 - dy)),
            int(min(W, x1 + dx)), int(min(H, y1 + dy)))


def ocen(mozaika, oryginal, prawda=None, outdir=None, max_wymiar=1600):
    # Gdy znamy prawde, dopasowujemy tylko do przeleconego fragmentu.
    # Na calym oryginale powtarzalne rzedy uprawy daja mnostwo niejednoznacznych
    # deskryptorow i test proporcji odrzuca prawie wszystkie pary.
    ox, oy = 0, 0
    wycinek = oryginal
    if prawda and prawda.get("kadry"):
        x0, y0, x1, y1 = _obszar_lotu(prawda, oryginal.shape[1], oryginal.shape[0])
        wycinek = oryginal[y0:y1, x0:x1]
        ox, oy = x0, y0

    gm, km = _szary(mozaika, max_wymiar)
    go, ko = _szary(wycinek, max_wymiar)

    # Najpierw SIFT (odporny na lekkie rozmycie mozaiki od mieszania kadrow),
    # a gdy znajdzie za malo par - ORB o niskim progu, ktory na drobnej
    # teksturze gleby znajduje wielokrotnie wiecej punktow.
    pary = kp1 = kp2 = None
    for det, norm in ((cv2.SIFT_create(nfeatures=8000), cv2.NORM_L2),
                      (cv2.ORB_create(nfeatures=8000, fastThreshold=5), cv2.NORM_HAMMING)):
        k1, d1 = det.detectAndCompute(gm, None)
        k2, d2 = det.detectAndCompute(go, None)
        if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
            continue
        p = _pary_wzajemne(d1, d2, norm)
        if pary is None or len(p) > len(pary):
            pary, kp1, kp2 = p, k1, k2
        if len(pary) >= 40:
            break
    if pary is None:
        return {"ok": False, "powod": "brak cech na obrazach"}
    if len(pary) < 12:
        return {"ok": False, "powod": f"za malo dopasowan ({len(pary)})"}

    # Wspolrzedne z powrotem do pelnej rozdzielczosci (/k), a punkty oryginalu
    # przesuniete o poczatek wycinka (ox, oy) - bo szukalismy tylko w obszarze lotu.
    src = np.float32([kp1[m.queryIdx].pt for m in pary]) / km
    dst = np.float32([kp2[m.trainIdx].pt for m in pary]) / ko + np.float32([ox, oy])
    # przeksztalcenie mozaika -> oryginal
    A, maska = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=4.0)
    if A is None:
        return {"ok": False, "powod": "nie udalo sie dopasowac mozaiki do oryginalu"}

    zgodne = int(maska.sum())
    skala = float(np.hypot(A[0, 0], A[1, 0]))
    # Skala bliska zera = RANSAC wybral model 'sciagnij wszystko w jeden punkt',
    # ktory formalnie pasuje do wielu blednych par. To nie jest prawdziwe dopasowanie.
    if skala < 0.2:
        return {"ok": False, "powod": "dopasowanie zdegenerowane - mozaika nie "
                "przypomina oryginalu", "zgodne_pary": zgodne}

    # mozaika przeniesiona do ukladu oryginalu
    H, W = oryginal.shape[:2]
    przen = cv2.warpAffine(mozaika, A, (W, H), flags=cv2.INTER_LINEAR)
    m_moz = cv2.warpAffine(np.full(mozaika.shape[:2], 255, np.uint8), A, (W, H),
                           flags=cv2.INTER_NEAREST) > 0
    # czarne tlo mozaiki (brak danych) nie liczy sie do porownania
    m_moz &= przen.sum(axis=2) > 0           # czarne tlo mozaiki to brak danych

    # obszar, ktory faktycznie zostal przeleciany - z prawdy
    if prawda:
        m_lot = np.zeros((H, W), np.uint8)
        for k in prawda["kadry"]:
            cv2.fillPoly(m_lot, [np.array(k["naroza"], np.int32)], 255)
        m_lot = m_lot > 0
    else:
        m_lot = np.ones((H, W), bool)

    wspolne = m_moz & m_lot
    pokrycie = float(wspolne.sum() / max(1, m_lot.sum()))

    # Roznica jasnosci piksel po pikselu, usredniona po 3 kanalach. int16, bo przy
    # uint8 odejmowanie 'zawinieloby sie' (np. 10 - 20 = 246).
    roznica = np.abs(przen.astype(np.int16) - oryginal.astype(np.int16)).mean(axis=2)
    blad = float(roznica[wspolne].mean()) if wspolne.any() else float("nan")
    mse = float((roznica[wspolne] ** 2).mean()) if wspolne.any() else float("nan")
    # PSNR = 10 * log10(MAX^2 / MSE). MAX = 255 dla 8 bitow. Kazde +6 dB to
    # mniej wiecej dwukrotnie mniejszy sredni blad.
    psnr = float(10 * np.log10(255 ** 2 / mse)) if mse and mse > 0 else float("inf")

    wynik = {
        "ok": True,
        "zgodne_pary": zgodne,
        "skala": round(skala, 4),
        "pokrycie": round(pokrycie, 4),
        "blad_sredni": round(blad, 2),
        "psnr_db": round(psnr, 2),
        "werdykt": werdykt(zgodne, skala, pokrycie, psnr),
    }

    if outdir:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # podglad: oryginal przygaszony, mozaika nalozona, roznica obok
        # podglad: przygaszony oryginal, na nim mozaika - od razu widac przesuniecia
        tlo = (oryginal * 0.35).astype(np.uint8)
        tlo[m_moz] = przen[m_moz]
        # mapa roznic w kolorach (niebieski = zgodnie, czerwony = duza roznica)
        mapa = cv2.applyColorMap(np.clip(roznica * 4, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        mapa[~wspolne] = 0
        cv2.imwrite(str(outdir / "ocena_nalozenie.jpg"), tlo, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(outdir / "ocena_roznica.jpg"), mapa, [cv2.IMWRITE_JPEG_QUALITY, 85])
        (outdir / "ocena.json").write_text(json.dumps(wynik, indent=2), encoding="utf-8")
    return wynik


def werdykt(zgodne, skala, pokrycie, psnr):
    """Werdykt opiera sie glownie na skali i PSNR. Liczba par mowi tylko,
    na ile ocena jest pewna - mala mozaika albo powtarzalne rzedy daja
    malo par nawet przy poprawnym zlozeniu."""
    if zgodne < 8 or not (0.85 < skala < 1.15):
        return "ZLE - mozaika nie odpowiada oryginalowi"
    dopisek = "  (niska pewnosc oceny - malo par)" if zgodne < 20 else ""
    if pokrycie < 0.7:
        return "CZESCIOWO - czesc kadrow nie trafila do mozaiki" + dopisek
    if psnr < 20:
        return "SLABO - widoczne przesuniecia lub zjawy" + dopisek
    if psnr < 25:
        return "DOBRZE - drobne niedokladnosci na szwach" + dopisek
    return "BARDZO DOBRZE" + dopisek
