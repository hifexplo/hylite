# hylite API Reference

## Class hierarchy

```
HyHeader (dict) — ENVI metadata
HyData — base [..., band]
├── HyImage     [x, y, band]
├── HyCloud     [point, band] + xyz, normals, rgb
├── HyLibrary   [sample, measurement, band]
└── Panel       calibration panel (correct module)

HyCollection (.hyc) — lazy disk-backed store
├── HyScene (.hys) — image + cloud + camera + PMap
└── MWL (.mwl)     — minimum wavelength results

HyFeature / MultiFeature / MixedFeature — absorption feature models
```

## HyData (base)

**Construction**: `HyData(data, header=None)`

### Indexing (`__getitem__` / `__setitem__`)

Prefer `[]` for header and band access on `HyImage`, `HyCloud`, and `HyLibrary`:

| Key | Behaviour |
|-----|-----------|
| `str` (alone) | ENVI/header field: `image['sensor']`, `image['wavelength']` |
| `float` | Single band by wavelength: `image[550.0]` → `get_band_index(550.0)` |
| `int` / `slice` on spatial axes | Direct `data` slice |
| Band axis: `float`, band-name `str`, or wavelength/name `slice` | Resolved via `get_band_index()` |
| `image[..., 680.0]` | Band at ~680 nm |
| `image[10, 20, 'Band 3']` | Pixel + band name |
| `image[..., 2000.0:2500.0]` | Wavelength range (inclusive stop) |
| `image['Band 1':'Band 50']` | Band-name range on band axis |

**Caveat**: bare `image['Band 3']` reads the header, not band 3 — use `image[..., 'Band 3']` or `image[550.0]`.

`HyLibrary`: sample/group names (`lib['EM1']`, `lib[['A','B']]`) take precedence; other strings fall through to header.

| Method | Purpose |
|--------|---------|
| `copy(data=True)` | Deep copy |
| `band_count()`, `samples()`, `lines()` | Dimensions |
| `get/set_wavelengths()`, `get_band_index(w)`, `get_band(b)` | Wavelength handling |
| `get/set_band_names()`, `get/set_fwhm()`, `get/set_bbl()` | Band metadata |
| `__getitem__(key)`, `__setitem__(key, value)` | Header + band-aware array indexing |
| `export_bands(range)`, `delete_nan_bands()`, `mask_bands()`, `mask_water_features()` | Band ops |
| `is_image()`, `is_point()`, `is_classification()` | Type checks |
| `X(onlyFinite)`, `get_raveled()`, `set_raveled()` | Flatten for ML |
| `resample(w)`, `smooth_median()`, `smooth_savgol()`, `fill_gaps()` | Spectral processing |
| `normalise()`, `percent_clip()`, `eval(op)` | Normalisation |
| `getQuantized()` / `fromQuanta()` | Lossy compression |
| `plot_spectra()`, `quick_plot(...)` | Visualisation |

## HyImage

Extends HyData. Shape `[x, y, band]`.

| Method | Purpose |
|--------|---------|
| `xdim()`, `ydim()`, `aspx()`, `T()` | Spatial dims |
| `set_projection()`, `set_projection_EPSG()`, `get_extent()` | Georeferencing |
| `pix_to_world()`, `world_to_pix()` | Coordinate transform |
| `crop()`, `resize()`, `tile()`, `mosaic()`, `flip()`, `rot90()` | Spatial ops |
| `mask()`, `crop_to_data()`, `drop_bbl()`, `fill_holes()`, `blur()`, `despeckle()` | Filtering |
| `pickPolygons()`, `pickPoints()`, `pickSamples()` | Interactive ROI |
| `get_keypoints()`, `match_keypoints()` | Feature matching |
| `createGIF()` | Animation |

## HyCloud

Extends HyData. Requires `xyz` (N×3).

| Method | Purpose |
|--------|---------|
| `point_count()`, `has_rgb/normals/bands()` | Properties |
| `set_bands()`, `add_bands()`, `delete_bands()` | Band management |
| `filter_points()`, `despeckle()`, `compute_normals(radius)` | Filtering |
| `project(image, cam, ...)` | Back-project image → cloud |
| `render(cam, bands)` | Cloud → image (rendering) |
| `quick_plot()`, `colourise()`, `plot_from_render()`, `mask()` | Visualisation |

## HyLibrary

Extends HyData. Shape `[sample, measurement, band]`.

| Method | Purpose |
|--------|---------|
| `sample_count()`, `get/set_sample_names()` | Sample metadata |
| `get_groups()`, `get_group()`, `add_group()` | Grouping |
| `merge()`, `collapse()`, `squash()`, `as_image()` | Combine/simplify |
| `__getitem__`, `__add__` | Indexing/concat |

**Module functions**: `from_indices(data, indices, s=4)`, `from_classification(data, labels, ...)`

## HyScene

Extends HyCollection (`.hys`).

| Method | Purpose |
|--------|---------|
| `construct(image, cloud, camera, s, occ_tol, maxf, bf, ...)` | Build PMap + geometry |
| `get_xyz/normals/depth/GSD/obliquity/view_dir/slope()` | Geometry accessors |
| `push_to_cloud(bands, method='best')` | Image → cloud |
| `push_to_image(bands, method='closest')` | Cloud → image |
| `match_colour_to(reference, method='norm'/'hist')` | Cross-scene matching |

## HyCollection

Out-of-core store (`.hyc`). Attributes accessed via `[]` (lazy load from disk):

```python
collection['image']       # get attribute by name
collection['reflectance'] = corrected
```

| Method | Purpose |
|--------|---------|
| `save()`, `save_attr()`, `get_file_dictionary()`, `clean()` | Persistence |
| `get/set/__getitem__/__setitem__`, `query()` | Access |
| `free()`, `free_attr()` | Memory management |
| `addExternal()`, `addSub()` | Nested/external refs |

`HyScene` (`.hys`) extends `HyCollection` — typical keys: `'image'`, `'cloud'`, `'camera'`, `'pmap'`.

## hylite.io

```python
io.load(path, to_nm=False)   # auto-dispatch
io.save(path, data, **kwds)
```

| Submodule | Key functions |
|-----------|---------------|
| `images` | `loadWithGDAL/Numpy/SPy`, `saveWithGDAL/Numpy/SPy`, `loadSubset` |
| `clouds` | `load/saveCloudPLY/LAS/CSV`, `loadCloudDEM` |
| `libraries` | `load/saveLibraryCSV/TXT/SED/TSG/LIB`, `loadLibraryDIR` |
| `pmaps` | `loadPMap`, `savePMap` |
| `cameras` | `loadCameraTXT`, `saveCameraTXT` |
| `headers` | `matchHeader`, `loadHeader`, `saveHeader` |

## hylite.correct

| Export | Module | Purpose |
|--------|--------|---------|
| `get_hull_corrected` | detrend | Convex hull continuum removal |
| `polynomial` | detrend | Polynomial detrend |
| `Panel` | panel | Calibration panel |
| `norm_eq`, `hist_eq` | equalize | Colour equalisation |

**correct.illumination** (import directly):

| Function | Purpose |
|----------|---------|
| `autoELC`, `UAC`, `ELC` | Empirical line / illumination correction |
| `IlluModel` | Forward radiance model |
| `estimate_illu`, `estimate_sun_vec`, `estimate_skyview` | Illumination estimation |
| `calcLambert`, `calcOrenNayar` | BRDF reflectance factors |
| `correct_path_absorption`, `estimate_path_radiance` | Atmospheric path |
| `calcBandRatioOcc`, `calcProjectedOcc` | Shadow/occlusion |

## hylite.analyse

| Module | Exports |
|--------|---------|
| `indices` | `band_ratio`, `NDVI`, `SKY`, `SHADE` |
| `mwl` | `minimum_wavelength(...)`, `MWL`, `colourise_mwl`, `plot_ternary`, `mwl_legend` |
| `unmixing` | `mix`, `unmix(method='nnls'/'fcls')`, `endmembers(method='nfindr'/'atgp'/...)` |
| `sam` | `spectral_angles`, `SAM` |
| `dtree` | `decision_tree` |
| `fourier` | `HyFourier`, `FourierArchive` — FFT compression, extrema, library search (`.fdr`/`.fda`) |

Removed in v1.4 (use [hklearn](https://github.com/samthiele/hklearn) instead): `supervised`, `unsupervised`.

### minimum_wavelength parameters

```python
minimum_wavelength(data, minw, maxw,
    trend='hull',      # detrend before fitting
    method='gauss',    # 'gauss'|'poly'|'quad'|'minmax'
    n=2,               # number of features to fit
    step=1,            # pixel subsampling
)
```

## hylite.filter (deprecated — do not use in new code)

**Deprecated.** Importing `hylite.filter` emits `DeprecationWarning`; the module will be removed. **Prefer `hylite.transform`** for PCA, MNF, and `overlay`.

| Export | Purpose |
|--------|---------|
| `PCA(data, bands, step, band_range)` | Legacy PCA (deprecated) |
| `MNF(data, bands, ...)` | Legacy MNF (deprecated) |
| `from_loadings(...)` | Apply existing loadings |

Removed in v1.4: `combine` → `transform.overlay`; `segment`, `label_blocks`, `TPT` → `transform.sample` / removed.

## hylite.transform

Most transforms do not require scikit-learn at import. Dimensionality reduction (`PCA`, `MNF`, `NoiseWhitener`) lives in **`hylite.transform.reduction`** (requires scikit-learn). These names are also available via lazy import from `hylite.transform` for backward compatibility.

| Export | Purpose |
|--------|---------|
| `convertToAbsorbance(data, method='kubelka-munk')` | Reflectance → absorbance |
| `overlay(image_list, method, warp=False)` | Multi-image fusion (replaces `filter.combine`) |
| `boost_saturation(...)` | HSV saturation boost |
| `Resample`, `ASTER`, `SENTINEL` | Spectral binning onto fixed band intervals |

### `hylite.transform.reduction` (requires scikit-learn)

| Export | Purpose |
|--------|---------|
| `NoiseWhitener` | Noise estimation for MNF |
| `MNF(n_components, noise=NoiseWhitener)` | MNF transformer |
| `PCA(n_components)` | PCA (MNF without whitening) |

### `Resample` (sensor band matching)

Average hyperspectral bands within `(min_wavelength, max_wavelength)` intervals. Band indices in `get_band()` are **1-based** (satellite convention). Wavelengths on output are interval midpoints.

```python
from hylite.transform import ASTER, SENTINEL, Resample

aster = ASTER.apply(image)              # HyImage, 14 bands
sentinel = SENTINEL.apply(image)        # HyImage, 13 bands
custom = Resample([(700., 900.), (2100., 2300.)])
binned = custom.apply(image)
band3 = ASTER.get_band(image, 3)        # single ASTER band as 2D array
```

**Sensor-to-sensor** — derive intervals from target `get_wavelengths()` and `get_fwhm()`:

```python
wav = fenix.get_wavelengths()
fwhm = fenix.get_fwhm() if fenix.has_fwhm() else np.abs(np.gradient(wav))
onto_fenix = Resample([(w - f / 2, w + f / 2) for w, f in zip(wav, fwhm)])
hyspex_on_fenix = onto_fenix.apply(hyspex)
```

**Note:** `HyData.resample(w)` selects nearest bands; `Resample.apply()` averages over intervals — use for ASTER/Sentinel bandpasses or matching another sensor's band definition.

## hylite.project

| Class/Function | Purpose |
|----------------|---------|
| `Camera(pos, ori, proj, fov, dims, step)` | Frame/pano camera |
| `Pushbroom(pos, ori, xfov, lfov, dims)` | Line scanner |
| `PMap(xdim, ydim, npoints)` | Sparse image↔point lookup |
| `proj_persp/pano/ortho`, `rasterize`, `pnp` | Projection geometry |
| `push_to_cloud/image`, `blend_scenes`, `get_blend_weights` | Data transfer |
| `project_pushbroom`, `optimize_boresight` | UAV pushbroom |
| `align`, `align_images`, `align_to_cloud`, `align_to_cloud_manual`, `refine_alignment`, `deepWarp`, `resample_raster` | Registration / warping |

## hylite.sensors

Base `Sensor`: `name()`, `fov()`, `correct_image()`, `correct_folder()`.

| Class | Sensor |
|-------|--------|
| `Fenix` | Specim Fenix / Fenix1k |
| `FX10`, `FX17`, `FX50` | Specim FX |
| `Rikola`, `Rikola_RSC1`, `Rikola_HSC2` | Rikola UAV |
| `OWL` | MWIR |
| `TelopsNano` | Telops Hypercam Nano |

`QAQC(image, method='LDPE'/'FT', dim=0/1/2)` — spectral accuracy vs reference.

## hylite.reference

| Module | Contents |
|--------|----------|
| `spectra` | `Target`, `loadTarget`, `R90`, `spectralon`, `PVC`, `custom` |
| `features` | `Features`, `Minerals`, `Themes` — predefined `HyFeature` instances |
| `generate` | `randomSpectra`, `genImage` — synthetic test data |

## File extensions

| Extension | Type |
|-----------|------|
| `.hdr` / `.dat` | HyImage (ENVI) |
| `.ply` / `.las` | HyCloud |
| `.csv` / `.lib` / `.sed` | HyLibrary |
| `.hyc` | HyCollection |
| `.hys` | HyScene |
| `.mwl` | MWL map |
| `.cam` / `.brm` | Camera / Pushbroom |
| `.npz` | PMap |
| `.fdr` | HyFourier archive |
| `.fda` | FourierArchive (multi-entry) |

## Test coverage map

| Test file | Covers |
|-----------|--------|
| `test_io.py` | load/save roundtrips |
| `test_core.py` | HyData, HyImage, HyCloud, HyLibrary basics |
| `test_transform.py` | PCA, MNF, overlay, resample |
| `test_correct.py` | hull, illumination, panel |
| `test_analyse.py` | MWL, Fourier, SAM, unmixing, indices |
| `test_project.py` | scene construct, push, align |
| `test_reference.py` | reference spectra/features |
| `test_sensors.py` | Fenix, FX correction |
| `test_multiprocessing.py` | split/merge HyData |

## Coregistration and warping (`hylite.project.align`)

| Function | Backend | Purpose |
|----------|---------|---------|
| `align(ref, mov, source_bands, dest_bands, method='affine', matchdist=0.6)` | OpenCV SIFT + skimage `warp` | Image-to-image; methods: `affine`, `piecewise_affine`, `polynomial` |
| `align_images(img1, img2, warp=True, **kwds)` | OpenCV homography/affine + optional `deepWarp` | **Deprecated**; use `align()` instead |
| `deepWarp(image, target)` | OpenCV DeepFlow + `remap` | Dense optical-flow warp between greyscale arrays |
| `align_to_cloud(image, cloud, cam, method='sift', recurse=2, sf=3)` | OpenCV SIFT/ORB + `pnp` | Auto-match rendered cloud RGB to HSI image |
| `align_to_cloud_manual(cloud, cam, points, pixels)` | OpenCV `pnp` | Manual ≥4 point–pixel pairs |
| `refine_alignment(image, cloud, cam, ...)` | SIFT + `pnp` | Refine pose using render at current estimate |
| `resample_raster(data, src_gt, dst_gt, dst_shape, order=1)` | GDAL `Warp` | Georeferenced raster resampling |

**HyImage helpers** (OpenCV): `get_keypoints(band, method='sift'|'orb')`, `match_keypoints(k1,k2,d1,d2, dist=0.7)`.

**Related**: `hylite.transform.overlay(..., warp=True)` — DeepFlow before stacking; `HyImage.mosaic()` — GDAL tile mosaicking; `Rikola.correct_image(align=True, warp='flow')` — intra-cube band alignment.
