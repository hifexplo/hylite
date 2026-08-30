# Literature and geological context

Background papers and datasets for hyperspectral geology workflows with hylite.

## Thiele et al. (2021) — Ore Geology Reviews, hylite introduction

DOI: [10.1016/j.oregeorev.2021.104252](https://doi.org/10.1016/j.oregeorev.2021.104252)

Presents **hylite** as open-source workflow for hypercloud creation. This **should be cited by codes using hylite**. Main workflow (Corta Atalaya VHMS open-pit, Spain):

1. Laboratory Fenix core-scanner → reflectance spectral library
2. Tripod Fenix panoramas → ELC + joint sun/sky illumination correction using photogrammetric geometry
3. UAV Rikola pushbroom → band alignment, Metashape camera poses, back-projection
4. Multi-sensor fusion (histogram equalisation, normal-weighted blending)
5. Analysis: MWL, spectral indices, Random Forest trained on lab spectra applied to outcrop hypercloud

**Primary citation** when using hylite.

## Thiele et al. (2021) — IEEE TGRS, illumination correction

DOI: [10.1109/TGRS.2021.3098725](https://doi.org/10.1109/TGRS.2021.3098725)

**Relevant for:** illumination correction (`IlluModel`, Oren–Nayar, skyview, sun vector); converting outcrop radiance to reflectance; topographic / joint sun–sky models; hypercloud radiometry.

Open-source illumination correction for hyperspectral digital outcrop models. Uses photogrammetric geometry (normals, occlusion, sky view) with sun and sky spectra to retrieve reflectance from at-sensor radiance.

## Thiele et al. (2022) — Remote Sensing, UAV SWIR mapping

DOI: [10.3390/rs14010005](https://doi.org/10.3390/rs14010005)

**Relevant for:** UAV / oblique SWIR; pushbroom geometry (`Pushbroom`, boresight); radiometric and geometric correction of airborne/UAV cubes; mineral mapping from corrected SWIR.

Mineralogical mapping with accurately corrected SWIR acquired obliquely from UAVs. Covers the geometric and radiometric steps needed before MWL or unmixing on UAV data.

## Lorenz et al. (2022) — Data, Black Angel dataset

DOI: [10.3390/data7080104](https://doi.org/10.3390/data7080104)

Km-scale open-access **hypercloud** of Zn–Pb mineralisation at Black Angel Mountain, Maarmorilik, Greenland. Documents a full open-source processing workflow. Dataset: https://rodare.hzdr.de/record/1642. Companion geology paper: Guarnieri et al. (2022) Minerals 12(7):800.

## Kirsch et al. (2023) — Photogrammetric Record, underground HSI

DOI: [10.1111/phor.12457](https://doi.org/10.1111/phor.12457)

Underground mine-face mapping at Zinnwald/Cínovec Sn-W-Li deposit. Uses hylite for: Fenix radiance correction, ELC reflectance, lens correction, multi-sensor co-registration via manual tie points, PnP camera pose, MWL and false-colour visualisation. Validates Li abundance maps against LIBS. Addresses illumination, moisture, and artificial lighting challenges.

## Chakraborty et al. (2024) — Remote Sensing, satellite HS comparison

DOI: [10.3390/rs16122089](https://doi.org/10.3390/rs16122089)

Compares PRISMA, EnMAP, EMIT vs airborne HyMap for carbonatite mapping in Namibia. Highlights SWIR inter-sensor inconsistency and need for accurate radiometric/topographic correction before geological mapping with satellite HS data.

## Thiele et al. (2024) — Frontiers in Earth Science, real-time drillcore

DOI: [10.3389/feart.2024.1433662](https://doi.org/10.3389/feart.2024.1433662)

**Relevant for:** drillcore scanning; real-time / on-site processing; crunchy pipelines; hycore sheds; hywiz viewing; MWL and band-ratio products; exploration logging and sampling decisions.

Open-source real-time workflow (crunchy + web viewer) for >Tb core-scan cubes. Applied to 6.4 km of core from Stonepark and Collinstown (Ireland) and Spremberg (Germany). Argues HS is most useful if acquired soon after drilling and processed on-site so results inform logging, sampling, and hole continuation.

## Chakraborty et al. (2024) — WHISPERS, airborne–satellite unmixing

DOI: [10.1109/WHISPERS65427.2024.10876472](https://doi.org/10.1109/WHISPERS65427.2024.10876472)

**Relevant for:** spectral unmixing (`unmix`, NNLS); mixed pixels; transferring endmembers from airborne (HySpex) to satellite (EnMAP); multi-scale coregistration; abundance mapping.

Selects endmembers on 2 m HySpex, NNLS-unmixes, resamples abundances to 30 m EnMAP, and predicts an EnMAP endmember library for wider-area unmixing. Improves on endmembers picked from EnMAP alone.

## Kamath et al. (2025) — Solid Earth, drillcore petrophysics

DOI: [10.5194/se-16-351-2025](https://doi.org/10.5194/se-16-351-2025)

**Relevant for:** predicting petrophysics from HS (density, slowness, gamma); CNN / deep learning on drillcore; hyperspectral upscaling; hycore; hklearn-style regression; downhole log resolution.

CNN maps millimetre-scale HS to density, slowness, and gamma-ray at Spremberg; tested on an independent borehole (R² ≈ 0.7–0.9 after resampling to log resolution). Shapley analysis for spectral drivers. Companion to Thiele et al. (2025) volcanic rock-property mapping.

## Thiele et al. (2025) — Solid Earth, volcanic rock properties

DOI: [10.5194/se-16-1249-2025](https://doi.org/10.5194/se-16-1249-2025)

**Relevant for:** predicting density, porosity, UCS, and Young's modulus from spectra; hydrothermal alteration; VNIR–LWIR feature importance; hklearn / MLP regression; volcanic geomechanics and hazard.

Lab VNIR–SWIR–MWIR–LWIR of basaltic–andesitic rocks from eight volcanoes. Nonlinear models (MLP) explain up to ~80 % of density/porosity variance and 65–70 % of UCS/E. Shapley values: VNIR–SWIR alteration features plus MWIR–LWIR glass/fabric/roughness.

## Thiele et al. (2026) — Minerals, hyperspectral geometallurgy

DOI: [10.3390/min16070674](https://doi.org/10.3390/min16070674)

**Relevant for:** modal mineralogy from HS; geometallurgy; SEM-MLA calibration; hklearn supervised models; hycore benchmark data; unbalanced / accessory-mineral prediction; drillcore mineral mapping.

Benchmarks SEM-MLA–coregistered VNIR–SWIR–MWIR–LWIR (~160k pixels, 200+ sections) to train supervised models that upscale mineral abundances along drillcores. Reasonable accuracy for rock-forming minerals; rare/accessory phases fail when training coverage is poor. Data/code: https://doi.org/10.14278/rodare.4582 (hycore sheds + hklearn notebooks).
