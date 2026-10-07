#!/usr/bin/env bash
# Symuluje naplyw zdjec w trakcie lotu - kopiuje pliki do katalogu
# obserwowanego w zadanych odstepach czasu.
#
# Uzycie (w dwoch terminalach):
#   terminal 1:  python -m stitchlive.cli watch dane/naplyw -o wyniki
#   terminal 2:  ./scripts/simulate_flight.sh /tmp/td/images dane/naplyw 2

set -euo pipefail
ZRODLO="${1:?podaj katalog zrodlowy}"
CEL="${2:?podaj katalog docelowy}"
ODSTEP="${3:-2}"

mkdir -p "$CEL"
echo "Symulacja lotu: $ZRODLO -> $CEL co ${ODSTEP}s"

for f in "$ZRODLO"/*.jpg "$ZRODLO"/*.JPG; do
  [[ -e "$f" ]] || continue
  # zapis do pliku tymczasowego i przeniesienie - operacja atomowa,
  # dzieki czemu obserwator nie zobaczy pliku w trakcie kopiowania
  cp "$f" "$CEL/.tmp_$(basename "$f")"
  mv "$CEL/.tmp_$(basename "$f")" "$CEL/$(basename "$f")"
  echo "  $(basename "$f")"
  sleep "$ODSTEP"
done
echo "Koniec symulacji."
