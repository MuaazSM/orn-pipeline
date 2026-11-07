# IMPACT implementation study

A study of **IMPACT** (Humbert-Vidan et al., *Image-based Mandibular and Maxillary Parcellation and
Annotation using Computed Tomography*, medRxiv 2025, doi:10.1101/2025.03.18.25324199): what its public
data and code contain, an exploratory analysis of all 60 CT scans, and a plan to reproduce the
segmentation models. **No model was trained.** The full write-up is the deck in
[`../docs/`](../docs/).

## Contents

| Path | What it is |
|---|---|
| `scripts/run_eda.sh` | Downloads the figshare release and runs the analysis end to end |
| `scripts/analyze.py` | Reads every CT and mask (numpy-only NIfTI reader): scan geometry, body-crop size, sub-volume presence and volume, tooth presence, dose per region |
| `scripts/figures.py` | CT overlays and front-view label projections for one patient |
| `results/results.json` | Per-patient measurements for all 60 scans and the 8 dose cases |
| `results/dose_summary.json` | Median mean/max dose per sub-volume and tooth group (7 usable cases) |
| `results/ct_headers.json` | CT dimensions and voxel spacing from the NIfTI headers |
| `figures/` | Figures used in the deck (patient 3) |

## Running it

```bash
bash impact_study/scripts/run_eda.sh
```

## Data

figshare: [doi:10.6084/m9.figshare.28615874](https://doi.org/10.6084/m9.figshare.28615874) (CC BY 4.0).
The CT data are not stored in this repository; the script downloads them into `impact_study/data/`.

Code from the authors: <https://github.com/dr-mnaser/orodental-segmentation>.

## Findings in brief

- 60 planning CTs, 512 × 512, slice thickness 1.0–3.3 mm (43 at 2.5 mm); every mask is on its CT's grid.
- All six mandible sub-volumes appear in every scan; maxilla basal sub-volumes appear in only 25–29 of 60.
- Median 27 of 32 teeth per patient; 42% of molar positions are empty.
- Dose to the mandible is strongly one-sided in some cases (up to 42 Gy between left and right basal bone).
- RTDOSE file for case 32 peaks at 16,384 (about twice the prescription), so it is excluded as a likely scaling error.
- The public code builds MONAI UNETR / residual UNet (the Swin UNETR import is commented out), uses 1,000
  epochs for teeth, and centres stage-2 crops on ground-truth masks; ensemble, test metrics and dosimetry are
  not included.
