"""Testy generatora realistycznego pola (scripts/generuj_pole.py)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

# skrypt nie jest czescia pakietu - wczytujemy go jako modul z pliku
_spec = importlib.util.spec_from_file_location("generuj_pole", ROOT / "scripts" / "generuj_pole.py")
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)


@pytest.fixture(scope="module")
def scena():
    # maly obraz, zeby test byl szybki
    return gp.generuj(900, 600, "marchew", gestosc_chwastow=2.0, seed=1)


def test_obraz_ma_wlasciwy_rozmiar_i_typ(scena):
    img, _ = scena
    assert img.shape == (600, 900, 3)
    assert img.dtype == np.uint8


def test_sa_chwasty_obu_klas(scena):
    _, et = scena
    klasy = {e["klasa"] for e in et}
    assert len(et) > 5
    assert klasy <= {0, 1}


def test_ramki_mieszcza_sie_w_obrazie(scena):
    _, et = scena
    for e in et:
        x0, y0, x1, y1 = e["xyxy"]
        assert 0 <= x0 < x1 <= 900
        assert 0 <= y0 < y1 <= 600


def test_te_same_ziarno_te_same_dane():
    a, ea = gp.generuj(400, 300, "burak", seed=5)
    b, eb = gp.generuj(400, 300, "burak", seed=5)
    assert np.array_equal(a, b)
    assert ea == eb


def test_wszystkie_uprawy_sie_generuja():
    for uprawa in gp.UPRAWY:
        img, _ = gp.generuj(500, 400, uprawa, seed=0)
        assert img.shape == (400, 500, 3)


def test_ramka_trafia_w_zielen(scena):
    """Srodek ramki chwastu ma byc zielony (roslina), a nie brazowy (gleba)."""
    img, et = scena
    trafione = 0
    for e in et:
        x0, y0, x1, y1 = e["xyxy"]
        b, g, r = img[int((y0 + y1) / 2), int((x0 + x1) / 2)].astype(int)
        trafione += g > r and g > b
    assert trafione / len(et) > 0.9


def test_zapis_etykiet_yolo(scena, tmp_path):
    img, et = scena
    gp.zapisz(img, et, tmp_path, "t")
    wiersze = (tmp_path / "labels" / "t.txt").read_text().strip().split("\n")
    assert len(wiersze) == len(et)
    for w in wiersze:
        klasa, cx, cy, sz, wy = w.split()
        assert klasa in ("0", "1")
        # format YOLO: wszystko znormalizowane do 0..1
        assert all(0 <= float(v) <= 1 for v in (cx, cy, sz, wy))
    assert (tmp_path / "images" / "t.jpg").exists()
    assert (tmp_path / "podglad" / "t.jpg").exists()


def test_yolo_odwraca_sie_na_piksele(scena, tmp_path):
    """Z pliku YOLO da sie odtworzyc oryginalne ramki w pikselach."""
    img, et = scena
    gp.zapisz(img, et, tmp_path, "t")
    h, w = img.shape[:2]
    for e, wiersz in zip(et, (tmp_path / "labels" / "t.txt").read_text().split("\n")):
        _, cx, cy, sz, wy = map(float, wiersz.split())
        x0, y0, x1, y1 = e["xyxy"]
        assert abs((cx - sz / 2) * w - x0) < 1.0
        assert abs((cy + wy / 2) * h - y1) < 1.0


def test_skrypt_z_wiersza_polecen(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "generuj_pole.py"), str(tmp_path),
                        "--liczba", "2", "--szer", "600", "--wys", "400", "--uprawa", "mix"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 2
    assert (tmp_path / "classes.txt").exists()
    assert (tmp_path / "opis.json").exists()
