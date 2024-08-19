#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
  echo "Korištenje skripte: ./prepare.sh [broj scenarija]"
  echo "Primjer: ./prepate.sh 1"
  exit 1
fi

SCENARIO_NUM="$1"
SCENARIO_DIR="podaci_link/Scenario${SCENARIO_NUM}"
OUT_FILE="podaci_prepare/data-Scenario${SCENARIO_NUM}.js"

set -e

if [[ ! -f "${SCENARIO_DIR}/STATIONS.txt" ]]; then
  echo "Datoteka '${SCENARIO_DIR}/STATIONS.txt' ne postoji."
  exit 1
fi

if [[ ! -f "${SCENARIO_DIR}/Scenario${SCENARIO_NUM}_data.zip" ]]; then
  echo "Datoteka '${SCENARIO_DIR}/Scenario${SCENARIO_NUM}_data.zip' ne postoji."
  exit 1
fi

if [[ ! -d "${SCENARIO_DIR}/figs" ]]; then
  echo "Direktorij '${SCENARIO_DIR}/figs' ne postoji."
  exit 1
fi

if [[ ! -d "${SCENARIO_DIR}/files" ]]; then
  echo "Direktorij '${SCENARIO_DIR}/files' ne postoji."
  exit 1
fi

for STATION_ID in $(sed -r 's/^([^,]+),([^,]+),([^,]+)\s+$/\1/' ${SCENARIO_DIR}/STATIONS.txt); do
  IMAGE_FILE_Z="${SCENARIO_DIR}/figs/Z/${STATION_ID}_Z.png"
  IMAGE_FILE_E="${SCENARIO_DIR}/figs/E/${STATION_ID}_E.png"
  IMAGE_FILE_N="${SCENARIO_DIR}/figs/N/${STATION_ID}_N.png"

  if [[ ! -f "$IMAGE_FILE_Z" ]]; then
    echo "Datoteka '${SCENARIO_DIR}/$IMAGE_FILE_Z' ne postoji."
    exit 1
  fi

  if [[ ! -f "$IMAGE_FILE_E" ]]; then
    echo "Datoteka '${SCENARIO_DIR}/$IMAGE_FILE_E' ne postoji."
    exit 1
  fi

  if [[ ! -f "$IMAGE_FILE_N" ]]; then
    echo "Datoteka '${SCENARIO_DIR}/$IMAGE_FILE_N' ne postoji."
    exit 1
  fi

  DATA_FILE="${SCENARIO_DIR}/files/${STATION_ID}.txt"

  if [[ ! -f "$DATA_FILE" ]]; then
    echo "Datoteka '$DATA_FILE' ne postoji."
    exit 1
  fi
done

echo "scenarios['scenario-${SCENARIO_NUM}'].points = [" > "$OUT_FILE"

sed -r 's|'\
'^([^,]+),([^,]+),([^,]+)\s+$'\
'|'\
'    { "id": "\1", "latitude": "\2", "longitude": "\3", "data": "files/\1.txt",'\
' "imageZ": "figs/Z/\1_Z.png", "imageE": "figs/E/\1_E.png", "imageN": "figs/N/\1_N.png" },|' \
"${SCENARIO_DIR}/STATIONS.txt" >> "$OUT_FILE"

echo "];" >> "$OUT_FILE"

