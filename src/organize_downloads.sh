#!/bin/bash
# Sort PeMS downloads into data/raw/. Run from the project root: bash src/organize_downloads.sh
set -e
SRC="$HOME/Downloads"
RAW="data/raw"
mkdir -p "$RAW/station_5min" "$RAW/meta" "$RAW/chp"

# Daily 5-minute traffic files
find "$SRC" -maxdepth 3 -type f -name 'd08_text_station_5min_*' -exec mv -n {} "$RAW/station_5min/" \;

# Station metadata
find "$SRC" -maxdepth 3 -type f -name 'd08_text_meta_*' -exec mv -n {} "$RAW/meta/" \;

# CHP zips: extract the files inside directly into chp/ (-j drops folders, -n skips existing)
find "$SRC" -maxdepth 3 -type f -name 'all_text_chp_*.zip' -exec unzip -n -j -q {} -d "$RAW/chp/" \;

# CHP files that macOS already unzipped
find "$SRC" "$RAW/station_5min" -maxdepth 3 -type f -name 'all_text_chp_*.txt.gz' -exec mv -n {} "$RAW/chp/" \;

# Remove empty leftover folders inside station_5min
find "$RAW/station_5min" -mindepth 1 -type d -empty -delete

echo "Done."
