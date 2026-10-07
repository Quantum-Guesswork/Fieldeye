"""Skladanie przyrostowy na cv2.detail.




Domyslny pipeline OpenCV zaklada kamere obracana wokol wlasnego srodka
(klasyczna panorama). Dron nad polem robi cos innego - PRZESUWA sie nad
plaska scena. Estymator rotacyjny probuje wtedy wyjasnic przesuniecie
obrotem, co daje absurdalne ogniskowe (rzedu kilkunastu tysiecy pikseli),
ogromne plotno i bardzo wolne renderowanie.

Wariant afiniczny (AffineBasedEstimator + BundleAdjusterAffinePartial +
warper "affine") modeluje przesuniecie, obrot w plaszczyznie i skale -
czyli dokladnie to, co zachodzi przy locie z nadiru.

To jest ten sam model, ktorego uzywa cv2.Stitcher w trybie SCANS, tylko
rozlozony na osobne bloki, zeby dalo sie go uzywac przyrostowo.

Pipeline:
    cechy -> dopasowanie par -> estymacja kamer -> dostrojenie wiazki
          -> wyrownanie ekspozycji -> szwy -> mieszanie
"""

import logging
import time
from pathlib import Path

import cv2
import numpy as np




class SkladaczDetail:
    """Przyrostowe skladanie z opcjonalnym oknem przesuwnym.

    okno=0  brak okna - kazda iteracja uwzglednia wszystkie dotychczasowe
            zdjecia. Wolniejsze, ale daje kompletna mozaike calej trasy.
    okno>0  tylko ostatnie N zdjec. Czas dodania jest w przyblizeniu staly,
            ale wynik pokrywa jedynie biezacy fragment trasy - przydatne
            jako podglad na zywo dla operatora, nie jako produkt koncowy.
    """

    # Parametry:
    #   okno         ile ostatnich zdjec brac do skladania (0 = wszystkie)
    #   conf_thresh  minimalna pewnosc polaczenia dwoch zdjec (liczona przez OpenCV
    #                z liczby zgodnych par); nizej = wiecej polaczen, ale mniej pewnych
    #   match_conf   prog testu proporcji w dopasowywaczu OpenCV
    #   model        "affine" (dron nad plaskim terenem) albo inny = rotacyjny (panorama)
    #   min_cech     kadr z mniejsza liczba punktow jest odrzucany
    def __init__(self, okno=0, work_megapix=0.6, conf_thresh=1.0,
                 match_conf=0.3, detektor="ORB", model="affine",
                 wyrownaj_ekspozycje=True, szwy=False, renderuj_co=1,
                 min_cech=40, zakres_par=0):
        self.okno = okno #liczba ostatnich zdj brancyh pod uwage pod zas skladania 0  oznacza uzycie wsyztkich
        self.work_megapix = work_megapix #skalowanie
        self.conf_thresh = conf_thresh #prog penwosci
        self.match_conf = match_conf #dopuszcanie tylko tych zdj ktore sa bardzeij pewno od poprzednich
        self.model = model #model affine jest dla drona na plaskim
        self.wyrownaj = wyrownaj_ekspozycje
        self.szwy = szwy #wlaczamy optymalne linie cieicia by bolo bardziej rowno
        self.renderuj_co = max(1, renderuj_co) # co ile odsiwezamy s treamlicie
        self.min_cech = min_cech #minimalna ilosc punktow charakterystycznych
        self.zakres_par = zakres_par #ogranicza dopasowanie do ostatnich zdj
        self.odrzucone = [] #lista zigonorowancyh

        self.finder = (cv2.ORB_create(nfeatures=2000) if detektor == "ORB"
                       else cv2.SIFT_create()) #inicjacja detektora szukajacyh punktow odniesienia

        self.sciezki = []
        self.cechy = []          # zbior wykreytycz pkt char dla kazdego obrazu
        self.robocze = []
        self.mozaika = None #wynik
        self.ostatni_czas = 0.0
        self.historia = []
        self._licznik = 0

    # ---------------- dobor blokow wg modelu ----------------

    # Trzy ponizsze metody dobieraja bloki pipeline'u OpenCV do modelu ruchu.
    # Model afiniczny = to samo, co cv2.Stitcher w trybie SCANS.
    def _matcher(self): #do parowania
        if self.model == "affine":
            return cv2.detail_AffineBestOf2NearestMatcher(False, False,
                                                          self.match_conf)
        return cv2.detail_BestOf2NearestMatcher(False, self.match_conf)

    def _estymator(self): #wyliacznie przesuniec i kadrow wzgl siebie
        if self.model == "affine":
            return cv2.detail_AffineBasedEstimator()
        return cv2.detail_HomographyBasedEstimator()

    def _dostrajacz(self): #wyrownaie
        if self.model == "affine":
            return cv2.detail_BundleAdjusterAffinePartial()
        return cv2.detail_BundleAdjusterRay()

    @property
    def _warper_typ(self): #rodzaj algorytmu rzutującego nanosi i przekszatlaca
        return "affine" if self.model == "affine" else "plane"

    # ---------------- dodawanie ----------------

    def _przeskaluj(self, img):
        piksele = img.shape[0] * img.shape[1]
        skala = min(1.0, np.sqrt(self.work_megapix * 1e6 / piksele))
        if skala < 1.0:
            img = cv2.resize(img, None, fx=skala, fy=skala,
                             interpolation=cv2.INTER_LINEAR_EXACT)
        return img

    def dodaj(self, sciezka):
        img = cv2.imread(str(sciezka)) #pobiera
        if img is None:
            log.warning("Nie mozna wczytac: %s", sciezka)
            return False

        roboczy = self._przeskaluj(img) #skaluje
        del img

        t0 = time.time()

        cechy = cv2.detail.computeImageFeatures2(self.finder, roboczy) #punkty charakterystyczne

        # kadr bez tekstury (rozmycie, jednolite pole) - odrzucamy zamiast
        # wysypac dopasowywacz, ktory potrzebuje co najmniej dwoch punktow
        if len(cechy.keypoints) < self.min_cech:
            log.warning("Pominieto %s: tylko %d punktow charakterystycznych",
                        Path(sciezka).name, len(cechy.keypoints))
            self.odrzucone.append(Path(sciezka).name)
            self._licznik += 1

            if self._licznik % self.renderuj_co == 0 and len(self.cechy) >= 2:
                return self._zloz()
            return False

        self.sciezki.append(sciezka)
        self.robocze.append(roboczy)
        self.cechy.append(cechy)
        self._licznik += 1


        if self.okno > 0 and len(self.robocze) > self.okno:
            for i in range(len(self.robocze) - self.okno):
                self.robocze[i] = None

        if len(self.sciezki) < 2:
            return False


        if self._licznik % self.renderuj_co != 0:
            return False

        ok = self._zloz()
        dt = time.time() - t0
        self.ostatni_czas = dt
        self.historia.append((len(self.sciezki), round(dt, 2), ok))
        return ok



    def _maska(self, numery):
        """Macierz, ktore pary zdjec dopasowywac.

        Bez ograniczenia dopasowujemy kazde z kazdym - n^2 par, przy stu
        zdjeciach to 10 000 dopasowan. Z zakres_par=N tylko zdjecia
        odlegle o najwyzej N w kolejnosci wykonania. N musi obejmowac
        sasiednia linie przelotu (~2x liczba zdjec w linii), inaczej
        linie nie zostana ze soba powiazane.
        """
        if not self.zakres_par:
            return None
        g = np.array(numery) #liczba wczytanycy zjdj
        return (np.abs(g[:, None] - g[None, :]) <= self.zakres_par).astype(np.uint8) #format pod open cv

    def _dopasuj(self, cechy, numery): #parujena podsawie maski
        matcher = self._matcher() #parowanie
        maska = self._maska(numery) #gotowa siacka
        # apply2 zwraca liste n*n obiektow MatchesInfo (kazda z kazda) z polem
        # confidence - na jej podstawie laczone sa zdjecia.
        pary = matcher.apply2(cechy) if maska is None else matcher.apply2(cechy, maska) #szuka powiazan
        matcher.collectGarbage() #zwalnia pamiec
        return pary

    def _zloz(self):
        od = max(0, len(self.sciezki) - self.okno) if self.okno > 0 else 0
        cechy = list(self.cechy[od:])
        obrazy = [o for o in self.robocze[od:] if o is not None]
        numery = list(range(od, len(self.cechy)))
        if len(obrazy) != len(cechy) or len(obrazy) < 2:
            return False

        try:
            pary = self._dopasuj(cechy, numery)
        except cv2.error as e:
            log.warning("Dopasowanie nieudane: %s", str(e).splitlines()[-1][:80])
            return False

        # odrzucenie zdjec niepowiazanych z reszta
        # Zostawia najwieksza grupe zdjec polaczonych z pewnoscia >= conf_thresh.
        # Zwraca numery zdjec, ktore zostaly.
        idx = cv2.detail.leaveBiggestComponent(cechy, pary, self.conf_thresh)
        idx = [int(i) for i in idx.ravel()] if idx is not None else []
        if len(idx) < 2:
            log.debug("Za malo powiazanych zdjec (%d z %d)", len(idx), len(cechy))
            return False

        if len(idx) != len(cechy):
            cechy = [cechy[i] for i in idx]
            obrazy = [obrazy[i] for i in idx]
            numery = [numery[i] for i in idx]
            pary = self._dopasuj(cechy, numery)

        # Wstepne polozenie kazdego zdjecia ('kamery') z par dopasowan - lancuchowo.
        ok, kamery = self._estymator().apply(cechy, pary, None)
        if not ok:
            log.debug("Estymacja kamer nieudana")
            return False
        for k in kamery:
            k.R = k.R.astype(np.float32)

        # Dostrojenie wiazki (bundle adjustment): poprawia polozenia wszystkich zdjec
        # naraz tak, zeby zminimalizowac bledy wszystkich par. Uwaga: przy kadrach
        # obroconych o kilka stopni globalna estymacja tego pipeline'u (estymator
        # + dostrojenie razem) daje rozmyta mozaike, choc same pary sa dokladne -
        # stad metoda global z wlasnym ukladem rownan.
        ba = self._dostrajacz()
        ba.setConfThresh(self.conf_thresh)
        ok, kamery = ba.apply(cechy, pary, kamery)
        if not ok:
            log.debug("Dostrojenie wiazki nieudane")
            return False

        # niepoprawna ogniskowa oznacza zle dopasowanie - lepiej odrzucic
        ogniskowe = [float(k.focal) for k in kamery]
        if not all(np.isfinite(f) and f > 0 for f in ogniskowe):
            log.debug("Niepoprawne ogniskowe: %s", ogniskowe)
            return False

        wynik = self._renderuj(obrazy, kamery, ogniskowe)
        if wynik is None:
            return False
        self.mozaika = wynik
        return True

    def _renderuj(self, obrazy, kamery, ogniskowe):
        # skala wyjsciowej mozaiki - mediana 'ogniskowych' wszystkich kamer
        skala = float(np.median(ogniskowe))
        warper = cv2.PyRotationWarper(self._warper_typ, skala)

        rogi, maski, przeksztalcone, rozmiary = [], [], [], []
        for img, k in zip(obrazy, kamery):
            K = k.K().astype(np.float32)
            R = k.R.astype(np.float32)
            try:
                rog, wimg = warper.warp(img, K, R, cv2.INTER_LINEAR,
                                        cv2.BORDER_REFLECT)
                maska = np.full(img.shape[:2], 255, np.uint8)
                _, wmask = warper.warp(maska, K, R, cv2.INTER_NEAREST,
                                       cv2.BORDER_CONSTANT)
            except cv2.error as e:
                log.debug("Warp nieudany: %s", e)
                return None

            rogi.append(rog)
            przeksztalcone.append(wimg)
            maski.append(wmask)
            rozmiary.append((wimg.shape[1], wimg.shape[0]))

        if self.wyrownaj:
            try:
                komp = cv2.detail_BlocksGainCompensator()
                komp.feed(rogi, przeksztalcone, maski)
                for i in range(len(przeksztalcone)):
                    komp.apply(i, rogi[i], przeksztalcone[i], maski[i])
            except cv2.error as e:
                log.debug("Wyrownanie ekspozycji pominiete: %s", e)

        if self.szwy and len(obrazy) > 1:
            try:
                finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
                znalezione = finder.find(
                    [p.astype(np.float32) for p in przeksztalcone], rogi, maski)
                maski = [m.get() if isinstance(m, cv2.UMat) else np.asarray(m)
                         for m in znalezione]
            except cv2.error as e:
                log.debug("Wyznaczanie szwow pominiete: %s", e)

        try:
            roi = cv2.detail.resultRoi(corners=rogi, sizes=rozmiary)
            blender = cv2.detail_MultiBandBlender()
            blender.prepare(roi)
            for wimg, maska, rog in zip(przeksztalcone, maski, rogi):
                blender.feed(wimg.astype(np.int16), maska, rog)
            wynik, _ = blender.blend(None, None)
        except cv2.error as e:
            log.debug("Mieszanie nieudane: %s", e)
            return None

        if wynik is None:
            return None
        return np.clip(wynik, 0, 255).astype(np.uint8)

    def raport_czasu(self):
        return [{"zdjec": n, "czas_s": t, "ok": ok} for n, t, ok in self.historia]
