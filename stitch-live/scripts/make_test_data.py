"""Generuje syntetyczne dane testowe: sztuczne pole pociete na zachodzace kadry.

Pozwala uruchomic i przetestowac caly program bez pobierania kilkuset
megabajtow danych i bez dostepu do drona.

Uzycie:
    python scripts/make_test_data.py dane/test --rows 3 --cols 6 --overlap 0.65
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stitchlive.pole import zbuduj_pole  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--rows", type=int, default=3, help="liczba linii przelotu")
    ap.add_argument("--cols", type=int, default=6, help="kadrow w linii")
    ap.add_argument("--tile", type=int, default=640, help="rozmiar kadru w px")
    ap.add_argument("--overlap", type=float, default=0.65,
                    help="pokrycie wzdluz linii (0-1)")
    ap.add_argument("--sidelap", type=float, default=0.5,
                    help="pokrycie miedzy liniami")
    ap.add_argument("--boustrophedon", action="store_true",
                    help="co druga linia w przeciwnym kierunku (jak realna misja)")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="losowe odchylenie pozycji kadru w px (symuluje dryf)")
    ap.add_argument("--exposure", type=float, default=0.04,
                    help="zmiennosc ekspozycji miedzy kadrami")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = Path(args.outdir)
    imgdir = out / "images"
    imgdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t = args.tile
    krok_x = int(t * (1 - args.overlap))
    krok_y = int(t * (1 - args.sidelap))
    pole_w = krok_x * (args.cols - 1) + t
    pole_h = krok_y * (args.rows - 1) + t

    pole = zbuduj_pole(pole_w, pole_h, seed=args.seed)
    cv2.imwrite(str(out / "pole_referencyjne.jpg"), pole,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    idx = 0
    for r in range(args.rows):
        kolumny = range(args.cols)
        if args.boustrophedon and r % 2 == 1:
            kolumny = range(args.cols - 1, -1, -1)

        for c in kolumny:
            idx += 1
            x0 = c * krok_x
            y0 = r * krok_y
            if args.jitter:
                x0 += int(rng.normal(0, args.jitter))
                y0 += int(rng.normal(0, args.jitter))
            x0 = max(0, min(x0, pole_w - t))
            y0 = max(0, min(y0, pole_h - t))

            kadr = pole[y0:y0 + t, x0:x0 + t].copy()

            if args.exposure:
                g = 1.0 + float(rng.normal(0, args.exposure))
                kadr = np.clip(kadr.astype(np.float32) * g, 0, 255).astype(np.uint8)

            cv2.imwrite(str(imgdir / f"IMG_{idx:04d}.jpg"), kadr,
                        [cv2.IMWRITE_JPEG_QUALITY, 94])

    print(f"Wygenerowano {idx} kadrow {t}x{t} px -> {imgdir}")
    print(f"Pole zrodlowe: {pole_w} x {pole_h} px -> {out/'pole_referencyjne.jpg'}")
    print(f"Pokrycie: {args.overlap:.0%} wzdluz linii, {args.sidelap:.0%} miedzy liniami")
    print()
    print("Uruchom skladanie:")
    print(f"  python -m stitchlive.cli batch {imgdir} -o wyniki")


if __name__ == "__main__":
    main()
