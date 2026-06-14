# Literature and geological context

Background papers and datasets for hyperspectral geology workflows with hylite.

## Thiele et al. (2021) — Ore Geology Reviews, hylite introduction

DOI: [10.1016/j.oregeorev.2021.104252](https://doi.org/10.1016/j.oregeorev.2021.104252)

Presents **hylite** as open-source workflow for hypercloud creation. Key workflow (Corta Atalaya VHMS open-pit, Spain):

1. Laboratory Fenix core-scanner → reflectance spectral library (grab-cut segmentation)
2. Tripod Fenix panoramas → ELC + joint sun/sky illumination correction using photogrammetric geometry
3. UAV Rikola pushbroom → band alignment, Metashape camera poses, back-projection
4. Multi-sensor fusion (histogram equalisation, normal-weighted blending)
5. Analysis: MWL, spectral indices, Random Forest trained on lab spectra applied to outcrop hypercloud

**Primary citation** when using hylite.

## Lorenz et al. (2022) — Data, Black Angel dataset

DOI: [10.3390/data7080104](https://doi.org/10.3390/data7080104)

Km-scale open-access **hypercloud** of Zn–Pb mineralisation at Black Angel Mountain, Maarmorilik, Greenland. Documents a full open-source processing workflow. Dataset: https://rodare.hzdr.de/record/1642. Companion geology paper: Guarnieri et al. (2022) Minerals 12(7):800.

## Kirsch et al. (2023) — Photogrammetric Record, underground HSI

DOI: [10.1111/phor.12457](https://doi.org/10.1111/phor.12457)

Underground mine-face mapping at Zinnwald/Cínovec Sn-W-Li deposit. Uses hylite for: Fenix radiance correction, ELC reflectance, lens correction, multi-sensor co-registration via manual tie points, PnP camera pose, MWL and false-colour visualisation. Validates Li abundance maps against LIBS. Addresses illumination, moisture, and artificial lighting challenges.

## Chakraborty et al. (2024) — Remote Sensing, satellite HS comparison

DOI: [10.3390/rs16122089](https://doi.org/10.3390/rs16122089)

Compares PRISMA, EnMAP, EMIT vs airborne HyMap for carbonatite mapping in Namibia. Highlights SWIR inter-sensor inconsistency and need for accurate radiometric/topographic correction before geological mapping with satellite HS data.

## Additional key papers (referenced in package)

- Illumination correction: Thiele et al. (2021) IEEE TGRS — [10.1109/TGRS.2021.3098725](https://doi.org/10.1109/TGRS.2021.3098725)
- UAV SWIR mapping: Thiele et al. (2022) Remote Sensing 14(1):5 — [10.3390/rs14010005](https://doi.org/10.3390/rs14010005)
