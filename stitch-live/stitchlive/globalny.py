"""Skladacz 'global' - pary z OpenCV, pozycje kadrow z jednego ukladu rownan.

Dlaczego osobna metoda
----------------------
Pipeline cv2.detail (i cv2.Stitcher w trybie SCANS) przy kadrach obroconych
wzgledem siebie o kilka stopni daje wyraznie rozmyta mozaike, mimo ze
pojedyncze pary kadrow dopasowuja sie z dokladnoscia subpikselowa.
Bez obrotu problem nie wystepuje - same przesuniecia sumuja sie poprawnie.

Tutaj pary liczy OpenCV (ORB + test proporcji + RANSAC), a polozenie
wszystkich kadrow wyznaczamy naraz, metoda najmniejszych kwadratow.

Model: przeksztalcenie podobienstwa (obrot + skala + przesuniecie)

    S(p) = [a -b] p + [tx]
           [b  a]     [ty]

jest LINIOWE w parametrach (a, b, tx, ty). Kazda para punktow p (kadr i)
i q (kadr j), ktore przedstawiaja ten sam punkt terenu, daje dwa rownania
liniowe S_i(p) - S_j(q) = 0. Kadr 0 ustalamy jako uklad odniesienia.
Rozwiazanie jest globalne - blad nie kumuluje sie wzdluz trasy.
"""

import logging
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)


# =====================================================================
# Glowna klasa metody 'global'. Uzycie:
#     s = SkladaczGlobalny()
#     for sciezka in zdjecia:
#         s.dodaj(sciezka)        # True, gdy mozaika zostala odswiezona
#     s.zakoncz()                 # koncowe renderowanie
#     s.mozaika                   # gotowy obraz (tablica BGR)
# =====================================================================
class SkladaczGlobalny:

    # Parametry:
    #   work_megapix     rozdzielczosc robocza w Mpx - kadry sa do niej zmniejszane
    #                    przed obliczeniami (mniej = szybciej, wiecej = dokladniej)
    #   zakres_par       z iloma poprzednimi kadrami dopasowywac nowy; 0 = ze wszystkimi
    #   min_zgodnych     minimalna liczba par punktow, zeby uznac dwa kadry za polaczone
    #   pkt_na_pare      ile punktow z kazdej pary trafia do ukladu rownan
    #   renderuj_co      co ile kadrow przeliczac i rysowac mozaike
    #   cechy            ile punktow charakterystycznych szukac na kadrze
    #   bloki_ekspozycji blokowe (dokladniejsze, duzo wolniejsze) wyrownanie jasnosci
    def __init__(self, work_megapix=0.6, zakres_par=0, min_zgodnych=15,
                 pkt_na_pare=40, wyrownaj_ekspozycje=True, szwy=False,
                 renderuj_co=1, detektor="ORB", okno=0, cechy=1500,
                 bloki_ekspozycji=False):
        self.work_megapix = work_megapix
        self.zakres_par = zakres_par
        self.min_zgodnych = min_zgodnych
        self.pkt_na_pare = pkt_na_pare
        self.wyrownaj = wyrownaj_ekspozycje
        self.szwy = szwy
        self.renderuj_co = max(1, renderuj_co)
        self.okno = okno
        self.bloki = bloki_ekspozycji

        # 1500 cech: dopasowanie brute force jest ~4x szybsze niz przy 3000,
        # a liczba par zgodnych praktycznie sie nie zmienia
        self.det = (cv2.ORB_create(nfeatures=cechy) if detektor == "ORB"
                    else cv2.SIFT_create(nfeatures=cechy))
        # zapasowy detektor o nizszym progu - dla kadrow rozmytych (drgania, ruch),
        # na ktorych domyslny prog FAST nie znajduje prawie niczego
        self.det_czuly = (cv2.ORB_create(nfeatures=cechy, fastThreshold=5) if detektor == "ORB"
                          else cv2.SIFT_create(nfeatures=cechy, contrastThreshold=0.01))
        # CLAHE: lokalne wyrownanie kontrastu w siatce 8x8 fragmentow. clipLimit
        # ogranicza wzmocnienie, zeby nie podbic szumu na jednolitej glebie.
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Miara podobienstwa deskryptorow: ORB daje deskryptory binarne (porownanie
        # = liczba roznych bitow, Hamming), SIFT - wektory liczb (odleglosc euklidesowa).
        self.norm = cv2.NORM_HAMMING if detektor == "ORB" else cv2.NORM_L2

        # Stan skladacza - rosnie z kazdym kadrem:
        #   obrazy  kadry w rozdzielczosci roboczej (potrzebne do renderowania)
        #   kp      wspolrzedne punktow charakterystycznych kazdego kadru
        #   des     ich deskryptory (do porownywania miedzy kadrami)
        self.sciezki, self.obrazy, self.kp, self.des = [], [], [], []
        self.pary = {}          # (i, j) -> (pkt_i, pkt_j) punkty zgodne
        self.S = None           # parametry podobienstwa, ksztalt (n, 4)
        self.mozaika = None
        self.historia = []
        self.ostatni_czas = 0.0
        # ile razy wywolano dodaj() - steruje odswiezaniem co renderuj_co
        self._licznik = 0
        # czy mozaika uwzglednia wszystkie dotychczasowe kadry
        self._aktualna = True

    # ---------------- dodawanie ----------------

    # Zmniejsza kadr tak, zeby mial okolo work_megapix milionow pikseli.
    # Pierwiastek, bo skala dziala na oba wymiary: 2x mniej w kazda strone = 4x mniej pikseli.
    def _przeskaluj(self, img):
        k = min(1.0, np.sqrt(self.work_megapix * 1e6 / (img.shape[0] * img.shape[1])))
        return img if k >= 1.0 else cv2.resize(img, None, fx=k, fy=k,
                                               interpolation=cv2.INTER_AREA)

    # Dodaje jeden kadr. Kolejnosc: wczytanie -> punkty charakterystyczne ->
    # dopasowanie do wczesniejszych kadrow -> (co renderuj_co) rozwiazanie ukladu
    # i narysowanie mozaiki. Zwraca True, gdy mozaika zostala odswiezona.
    def dodaj(self, sciezka):
        img = cv2.imread(str(sciezka))
        if img is None:
            log.warning("Nie mozna wczytac: %s", sciezka)
            return False
        img = self._przeskaluj(img)
        self._licznik += 1

        t0 = time.time()
        # lokalne wyrownanie kontrastu: winietowanie i zmiany ekspozycji przyciemniaja
        # krawedzie kadru - czyli dokladnie strefy nakladania - a prog FAST jest
        # bezwzgledny, wiec bez tego punkty znikaja tam, gdzie sa najbardziej potrzebne
        szary = self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        # detectAndCompute: jednoczesnie znajduje punkty (kp) i liczy ich deskryptory (des).
        kp, des = self.det.detectAndCompute(szary, None)
        # Za malo punktow - kadr prawdopodobnie rozmyty. Druga proba z nizszym progiem
        # detektora; bierzemy ja tylko, jesli faktycznie znalazla wiecej.
        if des is None or len(kp) < 200:
            kp2, des2 = self.det_czuly.detectAndCompute(szary, None)
            if des2 is not None and len(kp2) > (0 if kp is None else len(kp)):
                log.debug("%s: kadr malo ostry, prog obnizony (%d -> %d punktow)",
                          sciezka.name, 0 if kp is None else len(kp), len(kp2))
                kp, des = kp2, des2
        # Nawet po drugiej probie za malo - kadr jest bezuzyteczny, pomijamy go.
        if des is None or len(kp) < self.min_zgodnych:
            log.warning("Pominieto %s: za malo punktow (%d)", sciezka.name,
                        0 if kp is None else len(kp))
            return False

        n = len(self.obrazy)
        self.sciezki.append(sciezka)
        self.obrazy.append(img)
        # Z obiektow KeyPoint zostawiamy same wspolrzedne (x, y) - mniej pamieci.
        self.kp.append(np.float32([k.pt for k in kp]))
        self.des.append(des)

        # nowy kadr dopasowujemy tylko do poprzednich - kazda para liczona raz
        od = max(0, n - self.zakres_par) if self.zakres_par else 0
        nowe = 0
        for j in range(od, n):
            if self._dopasuj_pare(j, n):
                nowe += 1
        # Kadr bez zadnej pary nie trafi do mozaiki (chyba ze kolejne kadry go polacza).
        if n > 0 and nowe == 0:
            log.warning("%s: brak polaczenia z wczesniejszymi kadrami", sciezka.name)

        self._aktualna = False
        # Odswiezamy mozaike tylko co renderuj_co kadrow - samo dopasowanie par jest
        # tanie, rozwiazanie ukladu i renderowanie juz nie.
        if len(self.obrazy) < 2 or self._licznik % self.renderuj_co:
            return False

        ok = self._rozwiaz() and self._renderuj()
        self._aktualna = ok
        self.ostatni_czas = time.time() - t0
        self.historia.append((len(self.obrazy), round(self.ostatni_czas, 2), ok))
        return ok

    # Probuje dopasowac kadr i z kadrem j. Przy sukcesie zapisuje pary punktow
    # w self.pary[(i, j)] i zwraca True.
    def _dopasuj_pare(self, i, j):
        # Dla kazdego deskryptora z kadru i dwoch najpodobniejszych z kadru j
        # (BFMatcher = porownanie kazdy z kazdym, knn k=2 = dwoch najlepszych).
        m = cv2.BFMatcher(self.norm).knnMatch(self.des[i], self.des[j], k=2)
        # Test proporcji: para tylko wtedy, gdy najlepszy kandydat jest wyraznie
        # (o 25%) lepszy od drugiego. Odrzuca niejednoznaczne miejsca, np. powtarzalne rzedy.
        dobre = [a for a, b in (q for q in m if len(q) == 2) if a.distance < 0.75 * b.distance]
        if len(dobre) < self.min_zgodnych:
            return False

        # wspolrzedne dopasowanych punktow: p w kadrze i, q w kadrze j
        p = self.kp[i][[d.queryIdx for d in dobre]]
        q = self.kp[j][[d.trainIdx for d in dobre]]
        # RANSAC: szuka przeksztalcenia podobienstwa (obrot + skala + przesuniecie)
        # wspieranego przez jak najwiecej par. maska = ktore pary sie z nim zgadzaja
        # z dokladnoscia do 3 px (to one sa 'zgodne').
        E, maska = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                                               ransacReprojThreshold=3.0)
        if E is None:
            return False
        maska = maska.ravel().astype(bool)
        zgodne = int(maska.sum())

        # zabezpieczenia przed bledna para: za malo poparcia albo absurdalna skala
        # Skala z macierzy: E = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]], wiec
        # s = sqrt(E00^2 + E10^2). Dron nie zmienia wysokosci o 40% miedzy kadrami -
        # taka skala oznacza bledne dopasowanie.
        skala = float(np.hypot(E[0, 0], E[1, 0]))
        if zgodne < self.min_zgodnych or zgodne < 0.2 * len(dobre) or not 0.7 < skala < 1.4:
            return False

        # zostawiamy tylko pary zgodne z RANSAC
        p, q = p[maska], q[maska]
        if len(p) > self.pkt_na_pare:                  # rownomierna probka punktow
            wyb = np.linspace(0, len(p) - 1, self.pkt_na_pare).astype(int)
            p, q = p[wyb], q[wyb]
        self.pary[(i, j)] = (p, q)
        return True

    # ---------------- globalne wyrownanie ----------------

    # Graf: wezly = kadry, krawedzie = udane pary. Szukamy najwiekszej spojnej
    # skladowej przeszukiwaniem w glab (stos). Tylko te kadry da sie ze soba
    # powiazac - reszta nie ma wspolnego ukladu wspolrzednych.
    def _skladowa(self):
        """Najwieksza grupa kadrow polaczonych parami.

        Nie zakladamy, ze pierwszy kadr jest polaczony z reszta - bywa
        rozmyty albo zrobiony przy starcie. Uklad odniesienia to pierwszy
        kadr najwiekszej grupy.
        """
        n = len(self.obrazy)
        sasiedzi = {i: set() for i in range(n)}
        for i, j in self.pary:
            sasiedzi[i].add(j)
            sasiedzi[j].add(i)
        najlepsza, widziane = [], set()
        for start in range(n):
            if start in widziane:
                continue
            grupa, stos = {start}, [start]
            while stos:
                for x in sasiedzi[stos.pop()]:
                    if x not in grupa:
                        grupa.add(x)
                        stos.append(x)
            widziane |= grupa
            if len(grupa) > len(najlepsza):
                najlepsza = sorted(grupa)
        return najlepsza

    # Wyznacza polozenie wszystkich kadrow naraz. Wynik: self.S[k] = (a, b, tx, ty)
    # dla kazdego kadru k w mozaice.
    def _rozwiaz(self):
        """Uklad rownan liniowych: niewiadome (a, b, tx, ty) kazdego kadru."""
        wezly = self._skladowa()
        if len(wezly) < 2:
            return False
        # pozycja: numer kadru -> numer kolumny w ukladzie rownan. Pierwszy kadr
        # grupy dostaje pozycje 0 - to uklad odniesienia.
        pozycja = {k: i for i, k in enumerate(wezly)}
        pary = {k: v for k, v in self.pary.items() if k[0] in pozycja and k[1] in pozycja}

        for przebieg in range(3):          # odrzucanie par, ktore nie pasuja do reszty
            A, bvec = self._uklad(pary, pozycja, len(wezly))
            # Najmniejsze kwadraty: x minimalizuje ||A x - b||^2. Rownan (2 na kazda pare
            # punktow) jest duzo wiecej niz niewiadomych (4 na kadr).
            x, *_ = np.linalg.lstsq(A, bvec, rcond=None)
            # kadr odniesienia nie jest niewiadoma - dokladamy go jako tozsamosc
            S = np.vstack([[1.0, 0.0, 0.0, 0.0], x.reshape(-1, 4)])

            # Dla kazdej pary: jak daleko od siebie laduja jej punkty po przeksztalceniu.
            # Poprawna para - ulamek piksela; bledna (np. pomylony sasiedni rzad) - wiecej.
            bledy = {k: self._blad_pary(S[pozycja[k[0]]], S[pozycja[k[1]]], *v)
                     for k, v in pary.items()}
            zle = [k for k, e in bledy.items() if e > 3.0]
            if not zle:
                break
            log.debug("Odrzucam %d par o bledzie > 3 px", len(zle))
            for k in zle:
                pary.pop(k)
            # Po usunieciu par graf mogl sie rozpasc - wtedy przerywamy, zeby nie
            # zgubic kadrow z mozaiki.
            if len(self._skladowa_z(pary, wezly)) < len(wezly):
                break

        self.S = {k: S[pozycja[k]] for k in wezly}
        # mediana bledu par - miara jakosci dopasowania, trafia do raportu
        self.blad_px = float(np.median(list(bledy.values()))) if bledy else 0.0
        return True

    @staticmethod
    def _uklad(pary, pozycja, n):
        """Macierz ukladu rownan, budowana wektorowo.

        Dla punktu p z kadru i i q z kadru j:
          a_i*x1 - b_i*y1 + tx_i - (a_j*x2 - b_j*y2 + tx_j) = 0
          b_i*x1 + a_i*y1 + ty_i - (b_j*x2 + a_j*y2 + ty_j) = 0

        Kadr odniesienia (pozycja 0) ma parametry ZNANE (a=1, b=0, t=0),
        wiec nie jest niewiadoma - jego wyrazy przechodza na prawa strone.
        Kara za odchylenie od tozsamosci nie wystarcza: wspolczynniki rownan
        sa rzedu setek pikseli, wiec kadr 0 i tak by sie odchylal.
        """
        stale = np.array([1.0, 0.0, 0.0, 0.0])
        A_bl, b_bl = [], []
        for (i, j), (p, q) in pary.items():
            m = len(p)
            x1, y1, x2, y2 = p[:, 0], p[:, 1], q[:, 0], q[:, 1]
            # wspolczynniki przy (a, b, tx, ty) dla wiersza x i wiersza y
            Ci = np.empty((2 * m, 4))
            Ci[0::2] = np.column_stack([x1, -y1, np.ones(m), np.zeros(m)])
            Ci[1::2] = np.column_stack([y1, x1, np.zeros(m), np.ones(m)])
            Cj = np.empty((2 * m, 4))
            Cj[0::2] = -np.column_stack([x2, -y2, np.ones(m), np.zeros(m)])
            Cj[1::2] = -np.column_stack([y2, x2, np.zeros(m), np.ones(m)])

            # Blok macierzy dla tej pary: 2*m wierszy (x i y kazdego punktu),
            # 4*(n-1) kolumn (niewiadome wszystkich kadrow poza odniesieniem).
            B = np.zeros((2 * m, 4 * (n - 1)))
            b = np.zeros(2 * m)
            for C, k in ((Ci, pozycja[i]), (Cj, pozycja[j])):
                if k == 0:
                    b -= C @ stale                   # znany kadr -> prawa strona
                else:
                    B[:, 4 * (k - 1):4 * k] = C
            A_bl.append(B)
            b_bl.append(b)
        return np.vstack(A_bl), np.concatenate(b_bl)

    @staticmethod
    # Wersja _skladowa dla podanego zbioru par - do kontroli po odrzuceniu par.
    def _skladowa_z(pary, wezly):
        sasiedzi = {k: set() for k in wezly}
        for i, j in pary:
            sasiedzi[i].add(j)
            sasiedzi[j].add(i)
        widziane, stos = {wezly[0]}, [wezly[0]]
        while stos:
            for s in sasiedzi[stos.pop()]:
                if s not in widziane:
                    widziane.add(s)
                    stos.append(s)
        return widziane

    @staticmethod
    # Parametry (a, b, tx, ty) -> macierz 2x3 uzywana przez cv2.warpAffine.
    def _macierz(s):
        a, b, tx, ty = s
        return np.array([[a, -b, tx], [b, a, ty]], np.float64)

    # Oba zestawy punktow przeksztalcamy do ukladu mozaiki i mierzymy odleglosc.
    # Mediana zamiast sredniej - pojedynczy zly punkt jej nie zawyza.
    def _blad_pary(self, si, sj, p, q):
        Pi = np.hstack([p, np.ones((len(p), 1))]) @ self._macierz(si).T
        Pj = np.hstack([q, np.ones((len(q), 1))]) @ self._macierz(sj).T
        return float(np.median(np.linalg.norm(Pi - Pj, axis=1)))

    # ---------------- renderowanie ----------------

    # Rysuje mozaike z kadrow i ich parametrow self.S. Kazdy kadr jest
    # przeksztalcany tylko do prostokata, ktory zajmuje w mozaice (a nie do calego
    # plotna) - oszczednosc pamieci.
    def _renderuj(self):
        rogi, przeksztalcone, maski, rozmiary = [], [], [], []
        for k, s in self.S.items():
            img = self.obrazy[k]
            h, w = img.shape[:2]
            M = self._macierz(s)
            # Gdzie w mozaice laduja narozniki kadru - z nich obwiednia (x0, y0, x1, y1).
            nar = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ M.T
            x0, y0 = np.floor(nar.min(axis=0)).astype(int)
            x1, y1 = np.ceil(nar.max(axis=0)).astype(int)
            Mt = M.copy()
            # przesuniecie tak, zeby kadr trafil do swojego prostokata od (0, 0)
            Mt[0, 2] -= x0
            Mt[1, 2] -= y0
            rozm = (int(x1 - x0), int(y1 - y0))
            wimg = cv2.warpAffine(img, Mt, rozm, flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            # Maska: gdzie w prostokacie sa dane kadru (255), a gdzie puste rogi (0).
            wmask = cv2.warpAffine(np.full((h, w), 255, np.uint8), Mt, rozm,
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            rogi.append((int(x0), int(y0)))
            przeksztalcone.append(wimg)
            maski.append(wmask)
            rozmiary.append(rozm)

        if self.wyrownaj:
            try:
                # GainCompensator: jeden wspolczynnik na kadr - przy 48 kadrach
                # 0,8 s zamiast 52 s dla wersji blokowej, PSNR praktycznie ten sam
                komp = (cv2.detail_BlocksGainCompensator() if self.bloki
                        else cv2.detail_GainCompensator())
                komp.feed(rogi, przeksztalcone, maski)
                for i in range(len(przeksztalcone)):
                    komp.apply(i, rogi[i], przeksztalcone[i], maski[i])
            except cv2.error as e:
                log.debug("Wyrownanie ekspozycji pominiete: %s", e)

        if self.szwy and len(przeksztalcone) > 1:
            try:
                f = cv2.detail_GraphCutSeamFinder("COST_COLOR")
                znal = f.find([p.astype(np.float32) for p in przeksztalcone], rogi, maski)
                maski = [m.get() if isinstance(m, cv2.UMat) else np.asarray(m) for m in znal]
            except cv2.error as e:
                log.debug("Szwy pominiete: %s", e)

        # Mieszanie wielopasmowe: laczy kadry osobno dla drobnych i grubych
        # szczegolow, wiec szew jest gladki, a detale nie sa rozmyte.
        # prepare() dostaje obszar calej mozaiki, feed() kolejne kadry z ich polozeniem.
        blender = cv2.detail_MultiBandBlender()
        blender.prepare(cv2.detail.resultRoi(corners=rogi, sizes=rozmiary))
        for wimg, m, r in zip(przeksztalcone, maski, rogi):
            # blender wymaga liczb calkowitych ze znakiem (int16)
            blender.feed(wimg.astype(np.int16), m, r)
        wynik, _ = blender.blend(None, None)
        if wynik is None:
            return False
        self.mozaika = np.clip(wynik, 0, 255).astype(np.uint8)
        return True

    # Wywolac po ostatnim kadrze. Potrzebne, bo przy renderuj_co > 1 ostatnie
    # kadry moga jeszcze nie byc na mozaice.
    def zakoncz(self):
        """Koncowe renderowanie, jesli ostatnie kadry nie trafily do mozaiki."""
        if self._aktualna or len(self.obrazy) < 2:
            return self.mozaika is not None
        t0 = time.time()
        ok = self._rozwiaz() and self._renderuj()
        self.historia.append((len(self.obrazy), round(time.time() - t0, 2), ok))
        self._aktualna = ok
        return ok

    def raport(self):
        """Stan globalnego wyrownania - ile kadrow i par weszlo do mozaiki."""
        return {"kadrow_w_mozaice": len(self.S or {}),
                "kadrow_wczytanych": len(self.obrazy),
                "par": len(self.pary),
                "mediana_bledu_par_px": round(getattr(self, "blad_px", 0.0), 2)}

    def raport_czasu(self):
        return [{"zdjec": n, "czas_s": t, "ok": ok} for n, t, ok in self.historia]
