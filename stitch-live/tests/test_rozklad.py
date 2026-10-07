"""Testy rozkladu zdjecia na kadry i oceny mozaiki."""

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stitchlive.rozklad import (Ustawienia, rozloz, macierz_kadru,  # noqa: E402
                                naroza_kadru, wytnij, margines)
from stitchlive.ocena import ocen, werdykt                        # noqa: E402
from stitchlive.pole import zbuduj_pole                           # noqa: E402


@pytest.fixture(scope="module")
def pole():
    return zbuduj_pole(1400, 1000, seed=3)


# ---------------- geometria ----------------

def test_macierz_srodek_kadru_trafia_w_srodek_zadany():
    M = macierz_kadru(500, 300, 0, 400, 400)
    x, y = M @ np.array([200, 200, 1.0])
    assert abs(x - 500) < 1e-6 and abs(y - 300) < 1e-6


def test_macierz_obrot_zachowuje_srodek():
    M = macierz_kadru(500, 300, 37, 400, 400)
    x, y = M @ np.array([200, 200, 1.0])
    assert abs(x - 500) < 1e-6 and abs(y - 300) < 1e-6


def test_macierz_skala_zasiegu():
    """Kadr 400 px obejmujacy 800 px zrodla - bok w zrodle ma 800 px."""
    nar = naroza_kadru(macierz_kadru(1000, 1000, 0, 800, 400), 400)
    assert abs(math.dist(nar[0], nar[1]) - 800) < 1e-6


def test_wyciety_kadr_zgadza_sie_ze_zrodlem(pole):
    """Bez obrotu i skali kadr to dokladnie wycinek zrodla."""
    M = macierz_kadru(700, 500, 0, 400, 400)
    kadr = wytnij(pole, M, 400)
    wzor = pole[300:700, 500:900]
    assert np.abs(kadr.astype(int) - wzor.astype(int)).mean() < 2.0


def test_margines_rosnie_z_obrotem():
    bez = margines(400, Ustawienia())
    z = margines(400, Ustawienia(obrot=10))
    assert z > bez * 1.3


# ---------------- rozklad ----------------

def test_trasa_zapisuje_kadry_i_prawde(pole, tmp_path):
    p = rozloz(pole, tmp_path, Ustawienia(kadr=400, seed=1))
    pliki = sorted((tmp_path / "images").glob("*.jpg"))
    assert len(pliki) == len(p["kadry"]) >= 4
    assert (tmp_path / "prawda.json").exists()
    assert (tmp_path / "podglad_trasy.jpg").exists()
    assert cv2.imread(str(pliki[0])).shape[:2] == (400, 400)


def test_kadry_mieszcza_sie_w_zrodle(pole, tmp_path):
    p = rozloz(pole, tmp_path, Ustawienia(kadr=300, obrot=15, skala=0.1, seed=2))
    H, W = pole.shape[:2]
    for k in p["kadry"]:
        for x, y in k["naroza"]:
            assert -1 <= x <= W + 1 and -1 <= y <= H + 1


def test_losowy_zachowuje_pokrycie(pole, tmp_path):
    """Kolejne kadry leza blizej niz (1 - overlap) * zasieg."""
    u = Ustawienia(tryb="losowy", kadr=300, liczba=15, overlap=0.6, seed=4)
    p = rozloz(pole, tmp_path, u)
    srodki = [k["srodek"] for k in p["kadry"]]
    for a, b in zip(srodki, srodki[1:]):
        assert math.dist(a, b) <= 300 * (1 - 0.6) + 1


def test_ten_sam_seed_daje_to_samo(pole, tmp_path):
    a = rozloz(pole, tmp_path / "a", Ustawienia(tryb="losowy", kadr=300, liczba=8, seed=9))
    b = rozloz(pole, tmp_path / "b", Ustawienia(tryb="losowy", kadr=300, liczba=8, seed=9))
    assert [k["srodek"] for k in a["kadry"]] == [k["srodek"] for k in b["kadry"]]


def test_za_maly_obraz_zglasza_blad(tmp_path):
    with pytest.raises(ValueError):
        rozloz(np.zeros((200, 200, 3), np.uint8), tmp_path, Ustawienia(kadr=400))


def test_obrot_linii_daje_kursy_180(pole, tmp_path):
    p = rozloz(pole, tmp_path, Ustawienia(kadr=300, obrot_linii=True, seed=1))
    kursy = {round(k["kat_st"]) for k in p["kadry"]}
    assert kursy == {0, 180}


def test_na_zywo_kopiuje_bez_plikow_tymczasowych(pole, tmp_path):
    rozloz(pole, tmp_path / "s", Ustawienia(kadr=400, seed=1),
           na_zywo=tmp_path / "lot", odstep=0)
    pliki = list((tmp_path / "lot").iterdir())
    assert pliki and not any(p.name.startswith(".") for p in pliki)


# ---------------- ocena ----------------

def test_ocena_wycinka_oryginalu_jest_bardzo_dobra(pole):
    """Wycinek oryginalu to idealna 'mozaika' - ocena musi to rozpoznac."""
    w = ocen(pole[100:900, 100:1200].copy(), pole)
    assert w["ok"]
    assert abs(w["skala"] - 1.0) < 0.02
    assert w["psnr_db"] > 30


def test_ocena_odrzuca_zly_obraz(pole):
    smieci = np.random.default_rng(0).integers(0, 255, (600, 600, 3), dtype=np.uint8)
    w = ocen(smieci, pole)
    assert not w["ok"] or w["werdykt"].startswith("ZLE")


def test_werdykt_progi():
    assert werdykt(100, 1.0, 0.99, 30).startswith("BARDZO DOBRZE")
    assert werdykt(100, 1.5, 0.99, 30).startswith("ZLE")
    assert werdykt(100, 1.0, 0.5, 30).startswith("CZESCIOWO")
    assert "niska pewnosc" in werdykt(12, 1.0, 0.99, 30)


# ---------------- CLI ----------------

def test_cli_rozloz_bez_obrazu_generuje_pole(tmp_path):
    r = subprocess.run([sys.executable, "-m", "stitchlive.cli", "rozloz",
                        "--pole-w", "1200", "--pole-h", "900", "--kadr", "300",
                        "-o", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert list((tmp_path / "images").glob("*.jpg"))


def test_cli_demo_sklada_i_ocenia(tmp_path):
    r = subprocess.run([sys.executable, "-m", "stitchlive.cli", "demo",
                        "--pole-w", "1300", "--pole-h", "900", "--kadr", "360",
                        "-o", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
    w = json.loads((tmp_path / "ocena" / "ocena.json").read_text())
    assert w["ok"] and abs(w["skala"] - 1.0) < 0.05
