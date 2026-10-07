#!/usr/bin/env bash
# Pobiera prawdziwe zbiory zdjec lotniczych do testow skladania.
#
# Zbiory pochodza z repozytorium ODMdata (OpenDroneMap). Wszystkie maja
# realne pokrycie miedzy kadrami, wiec nadaja sie do testowania dopasowania
# na materiale, ktory nie zostal wygenerowany syntetycznie.
#
# Uzycie:
#   ./scripts/download_datasets.sh list
#   ./scripts/download_datasets.sh brighton_beach
#   ./scripts/download_datasets.sh mygla dane/

set -euo pipefail

DATASET="${1:-list}"
DEST="${2:-dane}"

declare -A REPOS=(
  [brighton_beach]="https://github.com/pierotofy/drone_dataset_brighton_beach.git"
  [mygla]="https://github.com/merkato/odm_mygla_dataset.git"
  [sheffield_park_3]="https://github.com/pierotofy/drone_dataset_sheffield_park_3.git"
  [sheffield_park_2]="https://github.com/pierotofy/drone_dataset_sheffield_park_2.git"
  [aukerman]="https://github.com/OpenDroneMap/odm_data_aukerman.git"
  [toledo]="https://github.com/OpenDroneMap/odm_data_toledo.git"
  [caliterra]="https://github.com/OpenDroneMap/odm_data_caliterra.git"
  [seneca]="https://github.com/OpenDroneMap/odm_data_seneca.git"
)

declare -A INFO=(
  [brighton_beach]="18 zdjec, ~62 MB  - NAJLEPSZY NA START, szybko sie sklada"
  [mygla]="29 zdjec, ~150 MB - dobry zbior startowy wg ODMdata"
  [sheffield_park_3]="32 zdjecia, ~165 MB"
  [sheffield_park_2]="54 zdjecia, ~283 MB"
  [caliterra]="77 zdjec, ~272 MB"
  [aukerman]="77 zdjec, ~543 MB - teren rolniczy, najblizej naszego zastosowania"
  [toledo]="87 zdjec, ~449 MB"
  [seneca]="167 zdjec, ~358 MB - do testow skalowania czasu"
)

if [[ "$DATASET" == "list" ]] || [[ -z "${REPOS[$DATASET]:-}" ]]; then
  echo "Dostepne zbiory:"
  echo
  for k in brighton_beach mygla sheffield_park_3 sheffield_park_2 caliterra aukerman toledo seneca; do
    printf "  %-18s %s\n" "$k" "${INFO[$k]}"
  done
  echo
  echo "Zrodlo: https://github.com/OpenDroneMap/ODMdata"
  [[ "$DATASET" == "list" ]] && exit 0 || exit 1
fi

mkdir -p "$DEST"
TARGET="$DEST/$DATASET"

if [[ -d "$TARGET" ]]; then
  echo "Katalog $TARGET juz istnieje - pomijam pobieranie."
else
  echo "Pobieram $DATASET (${INFO[$DATASET]})..."
  git clone --depth 1 "${REPOS[$DATASET]}" "$TARGET"
fi

if [[ -d "$TARGET/images" ]]; then
  IMGDIR="$TARGET/images"
else
  IMGDIR="$TARGET"
fi

COUNT=$(find "$IMGDIR" -maxdepth 1 -iname '*.jpg' | wc -l | tr -d ' ')
echo
echo "Gotowe. Zdjec: $COUNT"
echo "Katalog: $IMGDIR"
echo
echo "Sprawdz zbior:"
echo "  python -m stitchlive.cli info $IMGDIR"
echo
echo "Zloz mozaike (zacznij od kilku zdjec, pelne zdjecia sa duze):"
echo "  python -m stitchlive.cli batch $IMGDIR --limit 8 --megapix 0.3 -o wyniki"
