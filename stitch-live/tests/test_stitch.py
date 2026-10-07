"""Testy skladania na danych syntetycznych."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stitchlive.basic import SkladaczPodstawowy       # noqa: E402
from stitchlive.detail import SkladaczDetail          # noqa: E402
from stitchlive.source import lista_obrazow           # noqa: E402


@pytest.fixture(scope="module")
def dane(tmp_path_factory):
    """Maly zbior syntetyczny, generowany raz na caly modul."""
    out = tmp_path_factory.mktemp("dane")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_test_data.py"), str(out),
         "--rows", "2", "--cols", "4", "--tile", "400", "--overlap", "0.65"],
        check=True, capture_output=True)
    return out


@pytest.fixture(scope="module")
def sciezki(dane):
    return lista_obrazow(dane / "images")


def test_generator_tworzy_zdjecia(sciezki):
    assert len(sciezki) == 8
    img = cv2.imread(str(sciezki[0]))
    assert img.shape[:2] == (400, 400)


def test_lista_jest_posortowana(sciezki):
    nazwy = [p.name for p in sciezki]
    assert nazwy == sorted(nazwy)


# ---------------- skladacz podstawowy ----------------

def test_basic_sklada(sciezki):
    s = SkladaczPodstawowy(downscale=1.0)
    for p in sciezki:
        s.dodaj(p)
    assert s.mozaika is not None
    h, w = s.mozaika.shape[:2]
    # mozaika musi byc wieksza niz pojedynczy kadr
    assert w > 400 and h > 400


def test_basic_wymaga_dwoch_zdjec(sciezki):
    s = SkladaczPodstawowy()
    assert s.dodaj(sciezki[0]) is False
    assert s.mozaika is None


def test_basic_raportuje_czasy(sciezki):
    s = SkladaczPodstawowy(downscale=1.0)
    for p in sciezki[:4]:
        s.dodaj(p)
    r = s.raport_czasu()
    assert len(r) == 3                      # pierwsze zdjecie nie jest skladane
    assert all("czas_s" in x for x in r)


# ---------------- skladacz detail ----------------

def test_detail_sklada_wszystkie(sciezki):
    s = SkladaczDetail(okno=0, work_megapix=0.3)
    for p in sciezki:
        s.dodaj(p)
    assert s.mozaika is not None
    h, w = s.mozaika.shape[:2]
    assert w > 400 and h > 400


def test_detail_model_afiniczny_jest_domyslny():
    s = SkladaczDetail()
    assert s.model == "affine"
    assert s._warper_typ == "affine"


def test_detail_okno_ogranicza_pamiec(sciezki):
    """Obrazy spoza okna maja byc zwolnione."""
    s = SkladaczDetail(okno=3, work_megapix=0.3)
    for p in sciezki:
        s.dodaj(p)
    zwolnione = sum(1 for o in s.robocze if o is None)
    assert zwolnione == len(sciezki) - 3


def test_detail_okno_daje_staly_czas(sciezki):
    """Czas dodania kadru nie powinien rosnac wraz z dlugoscia serii."""
    s = SkladaczDetail(okno=4, work_megapix=0.3)
    for p in sciezki:
        s.dodaj(p)
    czasy = [t for _, t, ok in s.historia if ok]
    if len(czasy) >= 4:
        pierwsze = np.mean(czasy[:2])
        ostatnie = np.mean(czasy[-2:])
        # dopuszczamy trzykrotna rozbieznosc - chodzi o brak wzrostu liniowego
        assert ostatnie < pierwsze * 3 + 1.0


def test_detail_cechy_liczone_raz(sciezki):
    """Kazde zdjecie ma dokladnie jeden zestaw cech."""
    s = SkladaczDetail(okno=0, work_megapix=0.3)
    for p in sciezki:
        s.dodaj(p)
    assert len(s.cechy) == len(sciezki)


def test_detail_odrzuca_niepoprawny_obraz(tmp_path):
    s = SkladaczDetail()
    zly = tmp_path / "nieistniejacy.jpg"
    assert s.dodaj(zly) is False


def test_detail_renderuj_co_pomija_klatki(sciezki):
    s = SkladaczDetail(okno=0, work_megapix=0.3, renderuj_co=3)
    wyniki = [s.dodaj(p) for p in sciezki]
    # renderowanie tylko co trzecie zdjecie
    assert sum(wyniki) < len(sciezki) / 2


# ---------------- CLI ----------------

def test_cli_info(dane):
    r = subprocess.run(
        [sys.executable, "-m", "stitchlive.cli", "info", str(dane / "images")],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert "Obrazow:" in r.stdout


def test_cli_batch_tworzy_mozaike(dane, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "stitchlive.cli", "batch",
         str(dane / "images"), "-m", "detail", "--megapix", "0.3",
         "-o", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert (tmp_path / "mozaika_biezaca.jpg").exists()
    assert (tmp_path / "raport.json").exists()


def test_cli_basic_tworzy_mozaike(dane, tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "stitchlive.cli", "batch",
         str(dane / "images"), "-m", "basic", "--downscale", "1.0",
         "-o", str(tmp_path)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert (tmp_path / "mozaika_biezaca.jpg").exists()
