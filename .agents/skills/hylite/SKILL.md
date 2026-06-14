---
name: hylite
description: >-
  Open-source Python toolbox for hyperspectral geology — loading ENVI/PLY/LIB data,
  sensor correction, illumination correction, image-cloud projection (hyperclouds),
  MWL mapping, unmixing, and ML via hklearn (Stack/ModelSet, sklearn/PyTorch).
  Covers HyData, HyImage, HyCloud, HyLibrary, HyCollection, HyScene, ENVI file/header,
  get_band_index, and HyData [] indexing (wavelength and band-name slices).
  Use when working with hylite, hklearn, hyperspectral imaging, hyperclouds,
  digital outcrop models, mineral mapping, regression/classification on spectra,
  Fenix/Rikola/PRISMA/EnMAP data, or code in this repository.
---

# hylite

Open-source Python package for hyperspectral preprocessing, correction, 3-D projection, and geological analysis. Developed at HZDR-HIF (Helmholtz Institute Freiberg).

- **hylite** docs: https://hifexplo.github.io/hylite/hylite.html
- **hklearn** (companion ML package): https://github.com/samthiele/hklearn — depends on hylite; install with `pip install hklearn`
- **Literature / case studies**: [context.md](context.md)

## Workflow

```
Task progress:
- [ ] Step 1: Verify Python environment (if code will run)
- [ ] Step 2: Classify task type
- [ ] Step 3: Follow the task playbook
```

### Step 1: Python environment

Before running hylite related code or tests follow **[environment.md](environment.md)**.

Run the probe from the **skill directory** (`.agents/skills/hylite/`):

```bash
python scripts/check_env.py
```

From the **repo root**:

```bash
python .agents/skills/hylite/scripts/check_env.py
```

On `MISSING_DEPS`, stop and ask the user — do not `pip install` without consent. After `SUCCESS`, reuse the printed `PYTHON:` for the session.

**Skip the probe** for hyperspectral Q&A with no code.

**Install tiers** — `pip install hylite` installs the default scientific stack (numpy, scipy, matplotlib, scikit-learn, scikit-image, …). Optional extras:

| Extra | Packages | When needed |
|-------|----------|-------------|
| `[opencv]` | opencv-contrib-python | Image warping, coregistration, SIFT/ORB keypoints, optical flow (`align`, `deepWarp`, sensor band alignment) |
| `[gdal]` | GDAL | Georeferenced ENVI/GDAL I/O, `resample_raster`, `mosaic`, `io.loadWithGDAL` |
| `[all]` | opencv + gdal | Full geospatial + vision pipeline |
| `-e ".[all]"` | (this repo) | Local development against source |

Tests use tier simulation (`require_test_env(self, "opencv")` etc.) — skipped tests often mean a tier extra is not installed.

### Step 2: Classify the task

| Type | When | Playbook |
|------|------|----------|
| **Load data** | Load hyperspectral point clouds, images, libraries or some other hyperspectral format |  read `hylite.io.__init__.py` |
| **Correct or preprocess data** | Preprocessing and radiometric corrections of hyperspectral data | read py files in `hylite.sensors` and `hylite.correct` |
| **Geometric correction and projection** | Back-projection of data onto 3D point clouds, coregistration or geometric correction | read py files in `hylite.project` |
| **Data analysis** | Derive band ratios, minimum wavelength maps, unmixing, Fourier compression or other common hyperspectral analyses | Read py files in `hylite.analyse` and `hylite.transform` |
| **Machine learning** | Shallow or deep machine learning based classification or regression | See [hklearn-reference.md](hklearn-reference.md) |
| **Visualise or plot** | Create matplotlib-based visualisations of HSI data, spectra or results | See `quick_plot` and `plot_spectra` function of relevant core data class |
| **Hyperspectral Q&A** | Explain concepts, interpret spectra, no build requested | See core concepts and [context.md](context.md) |
| **Extend codebase** | Change hylite source, tests, or API | See [tasks/extend-codebase.md](tasks/extend-codebase.md) |

If ambiguous, ask one clarifying question. Many requests are **sequences** (e.g. preprocess → correct → analyse → visualise).

For all tasks requiring `hylite` code to be written or changed, see [reference.md](reference.md) for the API description.

## Core concepts

Hylite is built around the following core principles. These must be followed when implementing new code, and should also guide usage of the library.

**Polymorphism**: `HyImage`, `HyCloud`, and `HyLibrary` inherit from `HyData`. These are the core data classes used to store hyperspectral data. Most analysis functions (`minimum_wavelength`, hull correction, PCA) work on any of these types uniformly.

**Hypercloud**: A georeferenced point cloud where each point carries a reflectance spectrum — fusion of photogrammetric/LiDAR geometry with corrected hyperspectral imagery.

**Wavelengths**: Typically stored in nanometers. `io.load(path, to_nm=True)` converts on read. **Band selection accepts int (index), float (wavelength within `band_select_threshold=25` nm), or band name (string)** — via `get_band_index()` or `[]` indexing on `HyData` subclasses.

**Data layouts**:
- Images: `[x, y, band]`
- Clouds: `[point, band]` + separate `xyz` (N×3), optional `normals`, `rgb`
- Libraries: `[sample, measurement, band]`

**Indexing `HyData` (`HyImage`, `HyCloud`, `HyLibrary`)** — prefer `[]` for header and band access:

- **String alone** → ENVI/header field (`image['sensor']`, `image['wavelength']`). Not a band name — use `image[..., 'Band 3']` or `image[550.0]` for bands.
- **Float or band-name on band axis** → resolved via `get_band_index()`
- **Integer / slice on spatial axes** → direct array slicing

```python
image = io.load('scene.hdr')
cloud = io.load('hypercloud.ply')

# ENVI header / metadata
sensor = image['sensor']
wavelengths = image['wavelength']
image['description'] = 'Corta Atalaya pit'

# Single band by wavelength (nm) or band name
red   = image[..., 680.0]
band3 = image[10, 20, 'Band 3']
cloud[42, 2200.0]

# Band ranges — float wavelengths or band-name slices (inclusive stop)
swir = image[..., 2000.0:2500.0]
vnir = image['Band 1':'Band 50']
subset = cloud[..., 'Band 10':'Band 20']

# Spatial + spectral combined
patch = image[100:200, 50:150, 2140.0]
```

`HyLibrary` keeps sample/group selection (`lib['sample_A']`, `lib[['A', 'B']]`) and falls back to the above for header keys and array slicing (`lib[0, :, 550.0]`).

**HyCollection / HyScene** — lazy disk-backed stores (`.hyc`, `.hys`) expose attributes via `[]`:

```python
scene = HyScene('pit', './outputs/')
scene.construct(image, cloud, camera)
img = scene['image']      # HyImage
pc  = scene['cloud']      # HyCloud
cam = scene['camera']     # Camera
scene['reflectance'] = corrected_image
scene.save()
```

**Preset band combos** (from `import hylite`):
- `hylite.RGB` = (680, 550, 505)
- `hylite.SWIR` = (2200, 2250, 2350) — clay/mica/carbonate
- `hylite.VNIR`, `hylite.BROAD`, `hylite.MWIR`, `hylite.LWIR`

## Standard workflows

### 1. Load / save / inspect

```python
import hylite
from hylite import io

image = io.load('test_data/image.hdr')       # HyImage
cloud = io.load('test_data/hypercloud.ply')  # HyCloud
lib   = io.load('test_data/library.csv')     # HyLibrary

image.quick_plot(hylite.SWIR)
print(image['wavelength'][:5])
io.save('output.hdr', image)
```

`io.load` auto-detects type from extension (`.hdr`, `.ply`, `.las`, `.csv`, `.lib`, `.hyc`, `.hys`, `.mwl`, `.cam`).

### 2. Sensor preprocessing (Fenix example)

```python
from hylite.sensors import Fenix
from hylite.correct import Panel
from hylite.reference.spectra import R90
from hylite.correct.illumination import ELC

radiance = Fenix.correct_folder('/path/to/HIF_capture_dir')  # dark sub + lens + radiance
white_panel = Panel(R90, panel_radiance_spectra, wavelengths=radiance['wavelength'])
reflectance = ELC(radiance, white_panel)  # empirical line correction
```

Other sensors: `FX10`, `FX17`, `FX50`, `Rikola`, `OWL`, `TelopsNano`. Use `QAQC(image, method='LDPE')` for spectral accuracy checks.

### 3. Camera alignment, coregistration, and HyScene

#### Image ↔ point cloud (geometric registration)

Requires approximate camera pose; refines with OpenCV SIFT/ORB + PnP (`hylite.project.basic.pnp`). Needs **`hylite[opencv]`**.

```python
import numpy as np
from hylite import HyScene, io
from hylite.project import Camera, Pushbroom
from hylite.project.align import align_to_cloud_manual, align_to_cloud, refine_alignment

cam = Camera(np.zeros(3), np.zeros(3), 'pano', fov=32.3,
             dims=(image.xdim(), image.ydim()), step=0.084)
est, err = align_to_cloud_manual(cloud, cam, point_ids, pixel_coords)

est2, k3d, err = align_to_cloud(image, cloud, est, bands=hylite.RGB,
                                method='sift', sf=3, recurse=2, gf=True)
est3, src, k3d = refine_alignment(image, cloud, est2, method='sift', recurse=1)

S = HyScene('scene', './outputs/')
S.construct(image, cloud, est2, occ_tol=2.0, maxf=100, s=5)
reflectance_cloud = S.push_to_cloud(hylite.RGB, method='best')
S.save()

# Reload and access stored attributes
S2 = io.load('outputs/scene.hys')
hsi = S2['image']
```

Pushbroom UAV: `Pushbroom(positions, orientations, xfov, lfov, dims=...)` + `optimize_boresight()`.

Panoramic cameras need `cam.step` (angular pixel size). Pixel coords passed to PnP are converted from pixel centres (`- 0.5`).

#### Image ↔ image coregistration

**Preferred**: `align()` — SIFT matching + scikit-image warp (requires `scikit-image` and **`hylite[opencv]`**):

```python
from hylite.project.align import align

coreg = align(reference, moving, source_bands=hylite.RGB,
              dest_bands=hylite.RGB, method='affine', matchdist=0.6, vb=True)
```

Works best when images share sufficient spectral similarity; cross-range matching (RGB vs SWIR) is often difficult.

**Dense refinement** after affine — OpenCV DeepFlow (`deepWarp`):

```python
from hylite.project.align import deepWarp
import cv2

grey_ref = np.nanmean(reference[...], axis=-1)
grey_mov = np.nanmean(moving[...], axis=-1)
_, dmap = deepWarp(grey_mov, grey_ref)
refined = moving.copy(data=True)
for b in range(moving.band_count()):
    refined[..., b] = cv2.remap(refined[..., b], dmap, None, cv2.INTER_LINEAR)
```

Keypoint detection/matching on `HyImage` (used internally by `align`):

```python
k1, d1 = reference.get_keypoints(hylite.RGB, method='sift', mask=True)
k2, d2 = moving.get_keypoints(hylite.RGB, method='sift', mask=True)
src, dst = hylite.HyImage.match_keypoints(k1, k2, d1, d2, method='sift', dist=0.7)
```

#### Georeferenced resampling (GDAL)

Requires **`hylite[gdal]`**. Warp tiles in the same CRS with `resample_raster`:

```python
from hylite.project.align import resample_raster

resampled = resample_raster(image, src_gt=image.affine, dst_gt=out_affine,
                            dst_shape=(out_ydim, out_xdim), order=1)
```

`HyImage.mosaic(tiles, blend='mean', resampling='bilinear')` chains `resample_raster` across georeferenced tiles.

#### Stack / combine with optical-flow warp

```python
from hylite.transform import overlay

stacked, std = overlay([scan1, scan2, scan3], method='median', warp=True)
```

### 4. Illumination correction

```python
from hylite.correct.illumination import (
    estimate_sun_vec, estimate_skyview, calcLambert, calcOrenNayar,
    IlluModel, ELC, autoELC  # autoELC: fast but assumes brightest pixel = panel
)

sunvec, az, el = estimate_sun_vec(lat, lon, (date_str, fmt, tz))
skyview = estimate_skyview(normals)
rf = calcOrenNayar(normals, view, sunvec, roughness=0.2)
model = IlluModel(I=sun_spectra, S=sky_spectra, P=path_radiance,
                  skv=skyview, rf=rf, oc=occlusion)
radiance = model.getRadiance(reflectance)
```

Mask water bands (900–1000, 1100–1200, 1300–1550, 1750–2050 nm) for outdoor comparisons with lab spectra.

### 5. Spectral analysis

Use **`hylite.transform.reduction`** for PCA/MNF (not `hylite.filter` — deprecated). Other `hylite.transform` functions do not require scikit-learn at import.

```python
from hylite.correct import get_hull_corrected
from hylite.analyse import minimum_wavelength, colourise_mwl, band_ratio, SAM, unmix
from hylite.transform.reduction import PCA, MNF, NoiseWhitener
from hylite.reference.features import Minerals

hc = get_hull_corrected(data, vb=False)
swir_cube = hc[..., 2000.0:2500.0]

mwl = minimum_wavelength(cloud, 2140., 2380., trend='hull', method='gauss', n=4)
rgb_mwl = colourise_mwl(mwl)

pca = PCA(n_components=5).fit(image).transform(image)
mnf = MNF(n_components=5, noise=NoiseWhitener()).fit(image).transform(image)

amap = unmix(image, endmembers, method='nnls')
sam_map = SAM(image, reference_library)
feat = Minerals.get('AlOH')
```

**Resample onto satellite band schemes** — bin/average source bands within fixed wavelength intervals (`hylite.transform.sample.Resample`). Presets: `ASTER` (14 bands), `SENTINEL` (13 bands). Out-of-range intervals become `nan`.

```python
from hylite.transform import ASTER, SENTINEL, Resample

image = io.load('airborne.hdr', to_nm=True)

aster_img = ASTER.apply(image)        # 14 ASTER bands
sentinel_img = SENTINEL.apply(image)  # 13 Sentinel-2 bands
io.save('aster_sim.hdr', aster_img)

# Custom intervals: list of (min_wavelength, max_wavelength) tuples in nm
custom = Resample([(500.0, 600.0), (2000.0, 2500.0)])
binned = custom.apply(image)

# Extract one resampled band (1-based index, satellite convention)
swir1 = ASTER.get_band(image, 5)      # ASTER band 5 (~2165 nm mean)
ASTER.print_bands()                   # print interval per band
```

**Map one sensor onto another** — build `Resample` intervals from the target sensor's wavelength centres and FWHM (e.g. Hyspex → Fenix for cross-sensor comparison or library matching):

```python
import numpy as np
from hylite.transform import Resample

hyspex = io.load('hyspex_scene.hdr', to_nm=True)
fenix = io.load('fenix_scene.hdr', to_nm=True)

wav = fenix.get_wavelengths()
fwhm = fenix.get_fwhm() if fenix.has_fwhm() else np.abs(np.gradient(wav))
intervals = [(w - f / 2, w + f / 2) for w, f in zip(wav, fwhm)]

hyspex_on_fenix = Resample(intervals).apply(hyspex)
hyspex_on_fenix.set_band_names(fenix.get_band_names())
```

### 5b. Fourier compression and library search

```python
from hylite.analyse.fourier import HyFourier, FourierArchive

hyf = HyFourier(library, padding='cosine', max_freq=0.25)
hyf.save('library')  # writes library.fdr
loaded = HyFourier.load('library')
recon = loaded.toHyData()

archive = FourierArchive()
archive['sample_a'] = HyFourier(lib_a, max_freq=0.25)
archive.save('spectra')  # writes spectra.fda
```

### 6. Spectral libraries

```python
from hylite.hylibrary import from_classification, from_indices

cls = ...  # HyImage or array of class labels per pixel
lib = from_classification(image, cls, ignore=[0], subsample=100)
sample = lib['EM1']                    # single sample by name
swir_lib = lib[:, :, 2000.0:2500.0]    # wavelength slice
lib.merge(other_lib)
lib.resample(target_wavelengths)
```

USGS/other libraries: `io.load('spectra.sed')` or `hylite.io.libraries.loadLibrarySED`.

### 7. Machine learning with hklearn

**Prefer hklearn** for sklearn/PyTorch classification and regression. The removed `hylite.analyse.supervised` helpers are superseded by hklearn.

Two core classes: **`Stack`** (multi-sensor data + preprocessing + folds) and **`ModelSet`** (features + models + predict/ensemble).

See **[hklearn-reference.md](hklearn-reference.md)** for full workflow examples, inference, ensembles, and wavelength presets.

### 8. Multi-scene blending

```python
from hylite.project import get_blend_weights, blend_scenes
w = get_blend_weights([S1, S2], method='gsd', ascloud=True)
merged = blend_scenes([S1, S2], w, bands=(0, -1))
```

### 9. Satellite EO (PRISMA / EnMAP)

```python
image = io.load('satellite_L2_reflectance.hdr', to_nm=True)
image.crop(x0, y0, x1, y1)
image.mask_water_features()
image = get_hull_corrected(image)
vnir = image[..., 400.0:1000.0]
io.save('processed.hdr', image)
```

Use GDAL (`io.loadWithGDAL`) for georeferenced rasters when available.

## Conventions and pitfalls

- **Euler angles**: Same scheme as CloudCompare for camera orientation.
- **Compressed clouds and images**: Call `cloud.decompress()` or `image.decompress()` after loading integer-quantized `.ply` hyperclouds or `.hdr` (ENVI) files.
- **`hylite.filter` deprecated**: Importing `hylite.filter` emits `DeprecationWarning` and the module will be removed. **Use `hylite.transform`** for PCA, MNF, and `overlay`. Do not use `hylite.filter` in new code.
- **MWL speed**: `method='gauss'` best quality; `'poly'`/`'quad'` faster; `'minmax'` very fast, approximate.
- **autoELC**: Assumes brightest pixel is white panel — rough preview only.
- **Tests**: Run `pytest tests/` (hylite) and `pytest tests/` in hklearn repo.
- **hklearn inference**: `ModelSet.bind(new_stack)` replays training preprocessing when `stack.preprocessing` is empty (preferred).
- **Coregistration deps**: `scikit-image` for `align()`; OpenCV (`hylite[opencv]`) for SIFT, DeepFlow; GDAL (`hylite[gdal]`) for `resample_raster` / `mosaic`.
- **Band limits**: OpenCV `warpAffine` handles ≤512 bands per call; split cubes for larger band counts.

## API reference

- hylite: [reference.md](reference.md)
- hklearn: [hklearn-reference.md](hklearn-reference.md)
- literature: [context.md](context.md)

## When implementing new features

- Match existing patterns in sibling modules; reuse `HyData` methods rather than duplicating array logic.
- New sensors: subclass `hylite.sensors.Sensor`, add calibration data under `sensors/calibration_data/`.
- New I/O: extend `io.load`/`io.save` dispatch in `hylite/io/__init__.py`.
- Wavelength-aware band access via `[]` or `get_band_index()` (see **Indexing `HyData`** above).
- Keep it simple / clean and avoid adding lots of extra functions (including private `_functions`).
