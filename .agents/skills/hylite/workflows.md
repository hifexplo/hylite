# Standard hylite workflows

Read the matching source module first (`hylite.io`, `hylite.sensors`, …). These snippets are starting points, not a substitute for current APIs.

## 1. Load / save / inspect

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

`io.load` auto-detects type from extension (`.hdr`, `.ply`, `.las`, `.csv`, `.lib`, `.sed`, `.txt`, `.hyc`, `.hys`, `.mwl`, `.cam`). For Fourier archives use `HyFourier.load` / `FourierArchive.load` (`.fdr` / `.fda`), not `io.load`.

Partial ENVI reads (no full cube) go through `loadWithNumpy`:

```python
from hylite.io.images import loadWithNumpy

# keep listed bands (indices, wavelengths, or names)
swir = loadWithNumpy('scene.hdr', bands=[2000.0, 2200.0, 2350.0], to_nm=True)

# wavelength slice, or spatial + spectral stride
swir = loadWithNumpy('scene.hdr', bands=slice(2000.0, 2500.0))
preview = loadWithNumpy('scene.hdr', step=(2, 2, 2))          # every 2nd sample, line, band
binned = loadWithNumpy('scene.hdr', step=(2, 2, 2), average=True)  # block nan-mean

# named pixels → HyData spectra; memmap is a view (no copy)
spectra = loadWithNumpy('scene.hdr', pixels=[(10, 20), (30, 40)])
mm = loadWithNumpy('scene.hdr', memmap=True, mask_zero=False)
```

`loadSubset(path, bands=...)` / `loadSubset(path, pixels=...)` is a thin wrapper around `loadWithNumpy`. `average=True` cannot be combined with a band list.

## 2. Sensor preprocessing (Fenix example)

```python
from hylite.sensors import Fenix
from hylite.correct import Panel
from hylite.reference.spectra import R90
from hylite.correct.illumination import ELC

radiance = Fenix.correct_folder('/path/to/HIF_capture_dir')  # dark sub + lens + radiance
white_panel = Panel(R90, panel_radiance_spectra, wavelengths=radiance['wavelength'])
elc = ELC(white_panel)   # one Panel or a list of Panels
elc.apply(radiance)      # in-place; returns a boolean mask of well-corrected bands
```

Other sensors: `FX10`, `FX17`, `FX50`, `Rikola`, `OWL`, `TelopsNano`. Use `QAQC(image, method='LDPE')` for spectral accuracy checks.

## 3. Camera alignment, coregistration, and HyScene

### Image to point cloud

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

S2 = io.load('outputs/scene.hys')
hsi = S2['image']
```

Pushbroom UAV: `Pushbroom(positions, orientations, xfov, lfov, dims=...)` + `optimize_boresight()`.

Panoramic cameras need `cam.step` (angular pixel size). Pixel coords passed to PnP are converted from pixel centres (`- 0.5`).

### Image to image

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

### Georeferenced resampling (GDAL)

Requires **`hylite[gdal]`**. Warp tiles in the same CRS with `resample_raster`:

```python
from hylite.project.align import resample_raster

resampled = resample_raster(image, src_gt=image.affine, dst_gt=out_affine,
                            dst_shape=(out_ydim, out_xdim), order=1)
```

`HyImage.mosaic(tiles, blend='mean', resampling='bilinear')` chains `resample_raster` across georeferenced tiles.

### Stack / combine with optical-flow warp

```python
from hylite.transform import overlay

stacked, std = overlay([scan1, scan2, scan3], method='median', warp=True)
```

## 4. Illumination correction

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

## 5. Spectral analysis

Use **`hylite.transform.reduction`** for PCA/MNF (not `hylite.filter` — deprecated). Other `hylite.transform` functions do not require scikit-learn at import.

```python
from hylite.correct import get_hull_corrected
from hylite.analyse import minimum_wavelength, colourise_mwl, band_ratio, SAM, unmix
from hylite.transform.reduction import PCA, MNF, NoiseWhitener
from hylite.reference.features import Features

hc = get_hull_corrected(data, vb=False)
swir_cube = hc[..., 2000.0:2500.0]

mwl = minimum_wavelength(cloud, 2140., 2380., trend='hull', method='gauss', n=4)
rgb_mwl = colourise_mwl(mwl)

pca = PCA(n_components=5).fit(image).transform(image)
mnf = MNF(n_components=5, noise=NoiseWhitener()).fit(image).transform(image)

amap = unmix(image, endmembers, method='nnls')
sam_map = SAM(image, reference_library)
aloh = Features.AlOH  # list of HyFeature (approx. 2190 nm)
```

**Resample onto satellite band schemes** — bin/average source bands within fixed wavelength intervals (`hylite.transform.sample.Resample`). Presets: `ASTER` (14 bands), `SENTINEL` (13 bands). Out-of-range intervals become `nan`.

```python
from hylite.transform import ASTER, SENTINEL, Resample

image = io.load('airborne.hdr', to_nm=True)

aster_img = ASTER.apply(image)        # 14 ASTER bands
sentinel_img = SENTINEL.apply(image)  # 13 Sentinel-2 bands
io.save('aster_sim.hdr', aster_img)

custom = Resample([(500.0, 600.0), (2000.0, 2500.0)])
binned = custom.apply(image)

swir1 = ASTER.get_band(image, 5)      # ASTER band 5 (~2165 nm mean)
ASTER.print_bands()
```

**Map one sensor onto another** — intervals from the target sensor's wavelength centres and FWHM:

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

## 6. Fourier compression and library search

```python
from hylite.analyse.fourier import HyFourier, FourierArchive

hyf = HyFourier(library, padding='cosine', max_freq=0.25)
hyf.save('library')  # writes library.fdr
loaded = HyFourier.load('library')
recon = loaded.toHyData()

# Feature / name search (same syntax as ispec). Tokens AND; `|` ORs sub-queries.
names, scores = hyf.search('2200', confidence=10.0, n_result=10)  # absorption near 2200 nm
hyf.search('^2300')            # reflectance peak
hyf.search('!2200')            # absence of an absorption
hyf.search('2160-2200')        # feature anywhere in range
hyf.search('2200 Kaolinite')   # feature AND name tokens
hyf.search('kaolinite | dolomite')
hits = hyf.getSpectra(names)   # reconstruct matching spectra as HyLibrary

archive = FourierArchive()
archive['usgs'] = HyFourier(lib_a, max_freq=0.25)
archive.save('spectra')        # writes spectra.fda
loaded_arch = FourierArchive.load('spectra')
names, scores = loaded_arch.search('2200', n_result=10)
```

## 7. Spectral libraries

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

## 8. Machine learning with hklearn

Prefer hklearn for sklearn/PyTorch classification and regression. The removed `hylite.analyse.supervised` helpers are superseded by hklearn.

Two core classes: **`Stack`** (multi-sensor data + preprocessing + folds) and **`ModelSet`** (features + models + predict/ensemble).

See [hklearn-reference.md](hklearn-reference.md) for workflow examples, inference, ensembles, and wavelength presets.

## 9. Multi-scene blending

```python
from hylite.project import get_blend_weights, blend_scenes
w = get_blend_weights([S1, S2], method='gsd', ascloud=True)
merged = blend_scenes([S1, S2], w, bands=(0, -1))
```

## 10. Satellite EO (PRISMA / EnMAP)

```python
image = io.load('satellite_L2_reflectance.hdr', to_nm=True)
image = image.crop(xmin, xmax, ymin, ymax)  # pixel bounds; returns a copy
image.mask_water_features()
image = get_hull_corrected(image)
vnir = image[..., 400.0:1000.0]
io.save('processed.hdr', image)
```

Use GDAL (`io.loadWithGDAL`) for georeferenced rasters when available.
