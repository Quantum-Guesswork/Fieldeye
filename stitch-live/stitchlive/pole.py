"""Generator sztucznego pola - obraz zrodlowy do symulacji lotu."""

import cv2
import numpy as np


def zbuduj_pole(w, h, seed=7):
    """Sztuczna ortofotomapa pola: rzedy uprawy, gleba, chwasty, elementy terenu.

    Wazne dla testu skladania: musi miec wystarczajaco duzo lokalnie
    wyrozniajacych sie punktow, zeby detektor cech mial co dopasowywac.
    Jednolita tekstura dalaby zbior, na ktorym kazda metoda zawodzi.
    """
    rng = np.random.default_rng(seed)

    # gleba: brazowe tlo (BGR) z szumem, zeby nie byla jednolita
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = 70
    img[:, :, 1] = 95
    img[:, :, 2] = 125
    img = np.clip(img + rng.normal(0, 13, (h, w, 1)), 0, 255).astype(np.uint8)

    # rzedy uprawy
    # rzedy uprawy co 46 px, lekko nierowne jak po siewniku
    for x in range(0, w, 46):
        j = int(rng.integers(-3, 4))
        cv2.line(img, (x + j, 0), (x + j + int(rng.integers(-6, 7)), h),
                 (55, 140, 60), 13)

    # drobna tekstura roslin
    # drobne plamki tylko na rzedach (tam, gdzie kanal zielony jest jasny)
    for _ in range(w * h // 900):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if img[y, x, 1] > 110:
            cv2.circle(img, (x, y), int(rng.integers(2, 5)),
                       (40, int(rng.integers(120, 175)), 45), -1)

    # chwasty - jasniejsze skupiska
    # chwasty: jasniejsze kolka z obwodka
    for _ in range(90):
        x, y = int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20))
        r = int(rng.integers(7, 18))
        cv2.circle(img, (x, y), r, (60, 200, 90), -1)
        cv2.circle(img, (x, y), r, (45, 170, 70), 2)

    # elementy terenowe - kamienie, koleiny: punkty zaczepienia dla detektora
    for _ in range(70):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(img, (x, y), int(rng.integers(4, 11)), (110, 110, 115), -1)
    for _ in range(12):
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(img, (x1, y1), (x1 + int(rng.integers(-90, 90)),
                                 y1 + int(rng.integers(-90, 90))),
                 (95, 100, 110), int(rng.integers(2, 5)))
    return img
