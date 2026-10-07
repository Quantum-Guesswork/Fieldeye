"""Testy metody 'global' - na kadrach wycietych z jednego obrazu, ze znana prawda."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stitchlive.globalny import SkladaczGlobalny        # noqa: E402
from stitchlive.rozklad import Ustawienia, rozloz, margines  # noqa: E402
from stitchlive.pole import zbuduj_pole                 # noqa: E402
from stitchlive.ocena import ocen                       # noqa: E402


def _zloz(katalog, **kw):
    s = SkladaczGlobalny(work_megapix=0.3, **kw)
    for p in sorted(Path(katalog).glob("*.jpg")):
        s.dodaj(p)
    s.zakoncz()
    return s


@pytest.fixture(scope="module")
def pole():
    return zbuduj_pole(1600, 1100, seed=3)


@pytest.fixture(scope="module")
def obrocone(pole, tmp_path_factory):
    """Trasa z losowym obrotem kadrow - przypadek, ktory psuje cv2.detail."""
    out = tmp_path_factory.mktemp("obrot")
    prawda = rozloz(pole, out, Ustawienia(tryb="trasa", kadr=400, obrot=8, seed=5))
    return out, prawda


def test_sklada_kadry_obrocone(obrocone, pole):
    out, prawda = obrocone
    s = _zloz(out / "images")
    assert s.mozaika is not None
    assert len(s.S) == len(prawda["kadry"])          # wszystkie kadry w mozaice
    w = ocen(s.mozaika, pole, prawda)
    assert w["ok"] and w["psnr_db"] > 24


def test_pary_sa_dokladne(obrocone):
    out, _ = obrocone
    s = _zloz(out / "images")
    assert s.blad_px < 2.0                            # mediana bledu par w pikselach


def test_kadr_odniesienia_to_tozsamosc(obrocone):
    out, _ = obrocone
    s = _zloz(out / "images")
    np.testing.assert_allclose(s.S[0], [1, 0, 0, 0], atol=1e-3)


def test_losowe_ciecie_z_zakloceniami(pole, tmp_path):
    u = Ustawienia(tryb="losowy", kadr=400, liczba=15, obrot=5, skala=0.05,
                   rozmycie=0.3, szum=3, seed=11)
    prawda = rozloz(pole, tmp_path, u)
    s = _zloz(tmp_path / "images")
    assert len(s.S) >= 13                            # dopuszczamy zgubienie pojedynczych
    w = ocen(s.mozaika, pole, prawda)
    assert w["ok"] and w["psnr_db"] > 22


def test_rozmyty_kadr_nie_odpada(tmp_path, pole):
    """Zapasowy detektor ma uratowac kadr, na ktorym domyslny prog FAST nic nie widzi."""
    u = Ustawienia(tryb="trasa", kadr=400, rozmycie=1.0, seed=2)   # kazdy kadr rozmyty
    rozloz(pole, tmp_path, u)
    s = _zloz(tmp_path / "images")
    assert len(s.obrazy) == len(list((tmp_path / "images").glob("*.jpg")))


def test_zakoncz_renderuje_gdy_licznik_nie_doszedl(obrocone):
    out, _ = obrocone
    s = SkladaczGlobalny(work_megapix=0.3, renderuj_co=10_000)
    for p in sorted((out / "images").glob("*.jpg")):
        s.dodaj(p)
    assert s.mozaika is None                         # przy tak duzym progu - brak renderu
    assert s.zakoncz() and s.mozaika is not None


def test_zakres_par_ogranicza_dopasowania(obrocone):
    out, _ = obrocone
    pelny = _zloz(out / "images")
    ogr = _zloz(out / "images", zakres_par=3)
    assert all(abs(i - j) <= 3 for i, j in ogr.pary)
    assert len(ogr.pary) < len(pelny.pary)


def test_niepowiazany_kadr_nie_psuje_mozaiki(obrocone, tmp_path):
    out, _ = obrocone
    kat = tmp_path / "img"
    kat.mkdir()
    for p in sorted((out / "images").glob("*.jpg")):
        (kat / p.name).write_bytes(p.read_bytes())
    # obcy kadr - szum, bez zwiazku z polem
    obcy = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    cv2.imwrite(str(kat / "IMG_9999.jpg"), obcy)
    s = _zloz(kat)
    assert s.mozaika is not None
    obcy_idx = len(s.obrazy) - 1                     # obcy kadr jest ostatni
    assert obcy_idx not in s.S                       # nie trafil do mozaiki


def test_margines_nie_jest_przesadny():
    u = Ustawienia(obrot=2.0, skala=0.02)
    # przy malym obrocie i skali margines ma byc bliski polowie kadru,
    # a nie teoretycznego maksimum (sqrt(2) * 1.33)
    assert margines(640, u) < 640 / 2 * 1.25
