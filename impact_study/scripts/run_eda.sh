#!/bin/bash
# Download the IMPACT figshare release and run the exploratory analysis (no training).
# Needs ~6 GB free while downloading (2 GB kept afterwards). Python 3 with numpy, scipy, matplotlib.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/data/raw" && cd "$HERE/data"
get() { [ -s "raw/$1" ] || curl -sL --retry 3 -o "raw/$1" "https://ndownloader.figshare.com/files/$2"; }
get mandible.zip 53185970          # every CT + mandible masks (read in place by analyze.py)
get maxilla.zip 53186162
get teeth.zip 53186609
get dose_test_cases.zip 53185874
cd raw
unzip -q -o maxilla.zip '*/rs_mask_*' && unzip -q -o teeth.zip '*/rs_mask_*' && unzip -q -o dose_test_cases.zip
rm -f maxilla.zip teeth.zip        # their CTs duplicate mandible.zip
cd ..
python3 ../scripts/analyze.py      # -> results.json, figdata.npz
cp ../results/subvolume_names.json .
python3 ../scripts/figures.py 3    # -> fig_*_3.png (patient 3)
echo "Done. Outputs are in $HERE/data"
