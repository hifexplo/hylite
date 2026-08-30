---
name: hylite
description: >-
  Open-source Python toolbox for hyperspectral geology — loading ENVI/PLY/LIB data,
  sensor correction, illumination correction, image-cloud projection (hyperclouds), hyperspectral remote sensing,
  MWL mapping, unmixing, and ML via hklearn (Stack/ModelSet, sklearn/PyTorch).
  Covers HyData, HyImage, HyCloud, HyLibrary, HyCollection, HyScene, ENVI file/header,
  get_band_index, and HyData [] indexing (wavelength and band-name slices).
  Use when working with hylite, hklearn, hyperspectral imaging, hyperclouds,
  digital outcrop models, mineral mapping, regression/classification on spectra,
  Fenix/Rikola/PRISMA/EnMAP data, or code in this repository.
license: MIT
---

# hylite

Open-source Python package for hyperspectral preprocessing, correction, 3-D projection, and geological analysis. Developed by [Sam Thiele](https://www.samthiele.science/) at HZDR-HIF (Helmholtz Institute Freiberg).

- Docs: https://hifexplo.github.io/hylite/hylite.html
- Companion ML: [hklearn](https://github.com/samthiele/hklearn) (`pip install hklearn`)
- Literature: [context.md](context.md)

This skill follows the [Agent Skills](https://agentskills.io) format (YAML frontmatter + progressive disclosure). It is client-neutral: no Cursor-only or Claude-only tools.

## Workflow

```
Task progress:
- [ ] Step 1: Verify Python environment (if code will run)
- [ ] Step 2: Classify task type
- [ ] Step 3: Follow the task playbook
```

### Step 1: Python environment

Before running hylite code or tests, follow **[environment.md](environment.md)**.

From **this skill directory**:

```bash
python scripts/check_env.py
```

From the **hylite repo root**:

```bash
python .agents/skills/hylite/scripts/check_env.py
```

On `MISSING_DEPS`, stop and ask the user — do not `pip install` without consent. After `SUCCESS`, reuse the printed `PYTHON:` for the session.

**Skip the probe** for hyperspectral Q&A with no code.

**Install tiers** — `pip install hylite` includes the default scientific stack. Optional extras:

| Extra | Packages | When needed |
|-------|----------|-------------|
| `[opencv]` | opencv-contrib-python | Image warping, coregistration, SIFT/ORB, optical flow |
| `[gdal]` | GDAL | Georeferenced ENVI/GDAL I/O, `resample_raster`, `mosaic` |
| `[all]` | opencv + gdal | Full geospatial + vision pipeline |
| `-e ".[all]"` | (this repo) | Local development against source |

Tests use tier simulation (`require_test_env(self, "opencv")` etc.) — skipped tests often mean a tier extra is not installed.

### Step 2: Classify the task

| Type | When | Playbook |
|------|------|----------|
| **Load data** | Images, clouds, libraries, ENVI/PLY/LIB | Read `hylite/io/__init__.py` and `loadWithNumpy`; [workflows.md](workflows.md) §1 |
| **Correct or preprocess** | Radiometric / sensor correction | Read `hylite.sensors`, `hylite.correct`; [workflows.md](workflows.md) §2, §4 |
| **Geometry / projection** | Hyperclouds, coregistration, cameras | Read `hylite.project`; [workflows.md](workflows.md) §3, §9 |
| **Data analysis** | MWL, unmixing, Fourier, ratios | Read `hylite.analyse`, `hylite.transform`; [workflows.md](workflows.md) §5–6 |
| **Machine learning** | Classification / regression | [hklearn-reference.md](hklearn-reference.md); [workflows.md](workflows.md) §8 |
| **Visualise or plot** | Maps, spectra | `quick_plot` / `plot_spectra` on the relevant `HyData` class |
| **Hyperspectral Q&A** | Concepts, no code | Core concepts below; [context.md](context.md) |
| **Extend codebase** | Change source, tests, or API | [extend-codebase.md](extend-codebase.md) |

If ambiguous, ask one clarifying question. Many requests are sequences (preprocess → correct → analyse → visualise).

For code that uses or changes the API, also read [reference.md](reference.md). Prefer live source over this skill if they disagree.

## Core concepts

**Polymorphism**: `HyImage`, `HyCloud`, and `HyLibrary` inherit from `HyData`. Analysis functions (`minimum_wavelength`, hull correction, PCA) work on any of these types.

**Hypercloud**: A georeferenced point cloud where each point carries a reflectance spectrum.

**Wavelengths**: Typically nanometres. `io.load(path, to_nm=True)` converts on read. Band selection accepts **int** (index), **float** (wavelength within `band_select_threshold=25` nm), or **band name** — via `get_band_index()` or `[]`.

**Data layouts**:
- Images: `[x, y, band]`
- Clouds: `[point, band]` + `xyz` (N×3), optional `normals`, `rgb`
- Libraries: `[sample, measurement, band]`

**Indexing `HyData`** — prefer `[]` for header and band access:

- **String alone** → ENVI/header field (`image['sensor']`). Not a band name — use `image[..., 'Band 3']` or `image[550.0]`.
- **Float or band-name on the band axis** → `get_band_index()`
- **Integer / slice on spatial axes** → array slicing

```python
image = io.load('scene.hdr')
cloud = io.load('hypercloud.ply')

sensor = image['sensor']
wavelengths = image['wavelength']
image['description'] = 'Corta Atalaya pit'

red   = image[..., 680.0]
band3 = image[10, 20, 'Band 3']
cloud[42, 2200.0]

swir = image[..., 2000.0:2500.0]
vnir = image['Band 1':'Band 50']
subset = cloud[..., 'Band 10':'Band 20']
patch = image[100:200, 50:150, 2140.0]
```

`HyLibrary` keeps sample/group selection (`lib['sample_A']`, `lib[['A', 'B']]`) and falls back to the above (`lib[0, :, 550.0]`).

**HyCollection / HyScene** — lazy disk-backed stores (`.hyc`, `.hys`):

```python
scene = HyScene('pit', './outputs/')
scene.construct(image, cloud, camera)
img = scene['image']
pc  = scene['cloud']
cam = scene['camera']
scene['reflectance'] = corrected_image
scene.save()
```

**Preset band combos** (`import hylite`): `RGB` = (680, 550, 505); `SWIR` = (2200, 2250, 2350); also `VNIR`, `BROAD`, `MWIR`, `LWIR`.

## Conventions and pitfalls

- **Euler angles**: CloudCompare convention for camera orientation.
- **Compressed data**: call `decompress()` after loading integer-quantized `.ply` or ENVI.
- **`hylite.filter` is deprecated** — use `hylite.transform` for PCA, MNF, and `overlay`.
- **MWL speed**: `gauss` best; `poly`/`quad` faster; `minmax` approximate.
- **autoELC**: assumes brightest pixel is a white panel — preview only.
- **Tests**: `pytest tests/` in hylite; same in the hklearn repo.
- **hklearn inference**: `ModelSet.bind(new_stack)` replays training preprocessing when `stack.preprocessing` is empty.
- **Coregistration deps**: `scikit-image` for `align()`; OpenCV (`hylite[opencv]`) for SIFT/DeepFlow; GDAL (`hylite[gdal]`) for `resample_raster` / `mosaic`.
- **Band limits**: OpenCV `warpAffine` handles ≤512 bands per call; split larger cubes.

## Resources

Read only what the task needs (one hop from this file):

- Example recipes: [workflows.md](workflows.md)
- hylite API: [reference.md](reference.md)
- hklearn API: [hklearn-reference.md](hklearn-reference.md)
- Literature: [context.md](context.md)
- Editing this repo: [extend-codebase.md](extend-codebase.md)
- Environment: [environment.md](environment.md)
