# hklearn API Reference

Companion ML package for hylite. GitHub: https://github.com/samthiele/hklearn

**Dependencies**: numpy, scikit-learn, dill, hylite. Optional: PyTorch (MLP/CNN), joblib (parallel ensemble), shap (interpretability).

**Top-level exports**: `Stack`, `ModelSet`, wavelength presets `VNIR`, `SWIR`, `VSWIR`, `MWIR`, `LWIR`

## Architecture

```
HyImage / HyCloud / HyLibrary  (hylite)
         ↓
      Stack                     — flatten, mask NaNs, preprocess, transforms, folds
         ↓
     ModelSet                   — features + models + predict/ensemble
         ↓
   sklearn / PyTorch estimators
```

## Stack

Groups coregistered multi-sensor hyperspectral data and optional targets for ML.

**Construction**:
```python
Stack(names=['SWIR', 'LWIR'], data=[swir, lwir], y=y, target_names=['prop_A'])
```

| Method | Purpose |
|--------|---------|
| `X(sensor=None, transform=False, subset='all')` | Flattened feature matrix (n_samples × n_bands) |
| `y(name=None, transform=False, subset='all')` | Target vector/matrix |
| `reshape(data)` | Map flat predictions back to image/library shape |
| `mask(mask)` | Boolean mask on spatial/point dimension |
| `get_wavelengths(sensor=None)`, `get_sensors()` | Wavelength/sensor access |
| `set_sensor(name, data)`, `set_y(y, target_names)` | Update inputs |
| `print()` | Summary |

### Spectral preprocessing (chainable, stored in `stack.preprocessing`)

| Method | Purpose |
|--------|---------|
| `hc(ranges={}, hull={}, vb=True)` | Per-sensor hull correction via `hylite.correct.get_hull_corrected` |
| `smooth(window=7, method='savgol', order=1)` | Savitzky-Golay smoothing |
| `resample(wavelengths, bthresh=300.)` | Resample to standard bands; use `hklearn.VSWIR` etc. |
| `inv(clip=False)` | Invert spectra (absorptions → peaks) |
| `log(negative=True, ratio=False)` | Log / log-ratio transform |
| `norm()` | Per-spectrum normalisation |
| `subset(bands={})` | Band subset per sensor |

### Transforms and sampling

| Method | Purpose |
|--------|---------|
| `add_transform(name, transform)` | Fit sklearn `TransformerMixin` on `'y'` or sensor name |
| `pca(sensors=None, per_sensor=True, n_components=0.99)` | PCA per sensor |
| `set_groups(groups)`, `groups()`, `weights()` | Class balancing weights |
| `group_with_hdbscan(...)`, `group_with_binning(y, nbins=5)` | Auto grouping |
| `set_folds(k=5, use_groups=True)` | K-fold cross-validation masks in `stack.subsets` |
| `set_test_set(mask)`, `set_validation_set(mask)` | Hold-out splits |
| `stratified_sample(n)`, `random_sample(n)` | Subsample indices |

### Persistence

```python
stack.save('data.stack')
stack = Stack.load('data.stack')
```

Serialised with `dill`; includes preprocessing history and transforms.

## ModelSet

Manages feature extractors and prediction models bound to a Stack.

| Method | Purpose |
|--------|---------|
| `bind(stack)` | Attach stack; replay preprocessing if needed; copy transforms |
| `add_feature(name, model, xtransform=False)` | Add prefit feature extractor (must have `predict`) |
| `add_estimator(name, model, xtransform, features_in=False)` | Add prefit model |
| `fit_feature(name, model, target=0, xtransform=False, **kwds)` | Fit feature model (e.g. BRE, AbsorptionFeature) |
| `fit_model(name, model, xtransform=False, ytransform=False, force_no_features=False, **kwds)` | Fit predictor with optional GridSearchCV |
| `F(features=None, **kwargs)` | Concatenated feature matrix from all `features` |
| `predict(name=None, proba=False, **kwds)` | Single-model prediction |
| `predict_ensemble(models=None, combination_method='average', variance_metric=None, n_jobs=1)` | Multi-model ensemble + uncertainty |
| `get_score_table(...)` | Cross-validation / test metrics |
| `print()` | Summary |
| `save(path, stack=False)`, `ModelSet.load(path)` | Persistence |

### fit_model flags

- `xtransform=True`: apply Stack X-transforms (PCA etc.) before model input
- `ytransform=True`: apply Stack y-transforms (LogRatioScaler etc.)
- `features_in=True`: model input is `F()` not raw `X()`
- `force_no_features=True`: train on spectra even if features defined

## hklearn.estimators

| Class | Purpose |
|-------|---------|
| `BRE` | Band ratio estimator; wraps `hylite.analyse.band_ratio`; auto-fit bands via Lasso |
| `LME` | Linear mixture estimator: LMT unmixing + Lasso on abundances |
| `EnsembleEstimator` | Combine multiple fitted models; `average`/`median`/`mode`; variance via `std`/`iqr`/`entropy` |
| `MLP` | PyTorch MLP regressor/classifier; balanced batch sampling, optional PCA input |
| `CNN` | PyTorch 1D-CNN on spectra; similar API to MLP |
| `MHP` | Multi-hyperspectral phase estimator (compositional) |
| `Slice` | Slice-based estimator for spatial context |

**BRE example**:
```python
from hklearn.estimators import BRE
bre = BRE(wav=wavelengths, wav_range=(2000, 2500))  # auto-fit bands
M.fit_feature('aloh_bre', bre, target='AlOH', xtransform=True)
```

**LME example**:
```python
from hklearn.estimators import LME
lme = LME(breaks=[1000, 2500], wavelengths=wav)
M.fit_model('lme', lme, xtransform=True, ytransform=True)
```

## hklearn.transforms

All sklearn-compatible unless noted.

| Class/Function | Purpose |
|----------------|---------|
| `LogRatioScaler(base=0)` | ALR for compositional y (mineral abundances) |
| `Closure` | Close abundances to sum-to-one |
| `LMT` | Linear mixture transform (endmember unmixing features) |
| `AbsorptionFeature(features, wavelengths, depth=True, position=False)` | MWL-based handcrafted features via `hylite.analyse.minimum_wavelength` |
| `BoxCoxScaler` | Box-Cox on targets |
| `identity` | Pass-through; concat raw X with features in `F()` |
| `closure`, `ALR`, `iALR`, `CLR`, `iCLR` | Compositional data utilities |
| `mask_minor(X, athresh=0.1, cthresh=50)` | Drop accessory phases |

**AbsorptionFeature** — keys are feature names, values are kwargs for `minimum_wavelength`:
```python
features = {
    'F2200': dict(minw=2150., maxw=2240., method='poly', minima=True),
    'F2300': dict(minw=2250., maxw=2340., method='poly', minima=True),
}
T = AbsorptionFeature(features, wavelengths, depth=True)
```

## Workflow examples

### Training pipeline

```python
import hylite
import hklearn
from hklearn import Stack, ModelSet
from hklearn.transforms import LogRatioScaler, AbsorptionFeature
from hklearn.estimators import EnsembleEstimator
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression

# Targets as HyLibrary (samples × properties × 1)
y = hylite.HyLibrary(measured_props, lab=sample_ids, wav=[0, 1, 2, 3])
y.set_band_names(['density', 'porosity', 'UCS', 'E'])

S = Stack(['FENIX', 'FX50', 'LWIR'],
          data=[swir_img, mwir_img, lwir_img],
          y=y, target_names=['density', 'porosity', 'UCS', 'E'])

S = (S.smooth(window=5, order=1)
       .resample({'FENIX': hklearn.VSWIR, 'FX50': hklearn.MWIR, 'LWIR': hklearn.LWIR})
       .hc(hull={'FENIX': 'upper', 'FX50': 'upper', 'LWIR': 'lower'}, vb=False)
       .inv())

S.add_transform('y', LogRatioScaler(base=0))
S.pca(per_sensor=True, n_components=0.99)
S.set_folds(k=5, use_groups=True)

M = ModelSet().bind(S)
M.fit_model('svr', SVR(), xtransform=True, ytransform=True)
M.fit_model('pls', PLSRegression(n_components=5), xtransform=True)

F_SWIR = {'F2200': dict(minw=2150., maxw=2240., method='poly', minima=True)}
M.add_feature('fswir', AbsorptionFeature(F_SWIR, S.get_wavelengths('FENIX'), depth=True))
M.fit_model('svr_features', SVR(), features_in=True, xtransform=True)

S.save('MyDataStack.stack')
M.save('MyCoolModels.mset')
```

### Inference on new data

```python
M = ModelSet.load('ensemble.mset')
S2 = Stack(['FENIX', 'FX50', 'LWIR'], data=[swir_img, mwir_img, lwir_img])
S2.mask(background_mask)  # optional; call decompress() if integer-quantized
M.bind(S2)  # replays S.preprocessing + copies transforms
y_pred = M.predict(name='ensemble')
pred_image = hylite.HyImage(S2.reshape(y_pred))
pred_image.set_band_names(['density', 'porosity', 'UCS', 'E'])
```

### Ensemble + uncertainty

```python
est = EnsembleEstimator(M.models.values(), combination_method='average', n_jobs=4)
Me = ModelSet().bind(S)
Me.models['ensemble'] = est
y_mean, y_var = Me.predict_ensemble(combination_method='median', variance_metric='iqr')
```

**hklearn spectral resampling presets** (standardise models across sensors):
- `hklearn.VNIR`, `hklearn.SWIR`, `hklearn.VSWIR` (161 bands)
- `hklearn.MWIR` (149 bands), `hklearn.LWIR` (84 bands)

**PyTorch models** (optional): `hklearn.estimators.mlp.MLP`, `hklearn.estimators.cnn.CNN` — require PyTorch.

## Standard ML workflow patterns

### Regression (multi-property prediction)

1. Load assay CSV → `HyLibrary` targets
2. Load coregistered FENIX/FX50/LWIR → `Stack`
3. Preprocess: smooth → resample to hklearn presets → hull → inv
4. Transform y: `LogRatioScaler`, `RobustScaler` (sklearn)
5. `set_folds(5)` + optional spectral index features (BRE from hylite band_ratio)
6. `ModelSet.fit_model` for SVR, PLS, Ridge, LME
7. Build `EnsembleEstimator`, save `.set` file
8. Inference: new `Stack` per block → `bind` → `predict` → `reshape` → `HyImage`

### Classification

Same Stack/ModelSet pattern; use sklearn classifiers (`RandomForestClassifier`, etc.) with `fit_model(..., scoring='accuracy')`. Use `Stack.weights()` for imbalanced classes. `EnsembleEstimator(..., combination_method='mode'|'probability')`.

### Feature engineering with hylite

hklearn features often wrap hylite analysis:
- `BRE` → `hylite.analyse.band_ratio`
- `AbsorptionFeature` → `hylite.analyse.minimum_wavelength`
- `LME` / `LMT` → `hylite.analyse.unmix`
- Stack `.hc()` → `hylite.correct.get_hull_corrected`

Compute exploratory indices in hylite first, then pass as extra Stack sensors or ModelSet features.

## Wavelength presets

| Name | Range | Bands | Notes |
|------|-------|-------|-------|
| `VNIR` | 450–1000 nm, 20 nm | 27 | Ultra-blue removed |
| `SWIR` | 1000–2500 nm | 134 | 20 nm below 1300, 10 nm above |
| `VSWIR` | VNIR + SWIR | 161 | Common VSWIR sensors |
| `MWIR` | 3000–5250 nm, 15 nm | 149 | Below 3000 nm removed |
| `LWIR` | 7750–12000 nm, 50 nm | 84 | TIR noise edges trimmed |

SWIR/MWIR/LWIR/VNIR stored as `(start, end)` band pairs for resampling.

## Test coverage (hklearn repo)

| Test file | Covers |
|-----------|--------|
| `test_stack.py` | Stack X/y, mask, preprocess chain, PCA, folds, save/load |
| `test_ModelSet.py` | fit_model, predict, features, scores |
| `test_MLAStack.py` | Multi-stack workflows |
| `test_Transforms.py` | LogRatio, AbsorptionFeature, LMT |
| `test_estimators.py` | BRE, LME, ensemble |

## hylite ↔ hklearn integration checklist

1. Correct to reflectance in hylite before ML (ELC, illumination correction)
2. Coregister sensors (HyScene, alignment) before Stack
3. Use `HyLibrary` for training samples; `HyImage`/`HyCloud` for spatial prediction
4. Match preprocessing: training Stack preprocessing is replayed on inference via `ModelSet.bind`
5. Resample to `hklearn.VSWIR` (etc.) so models transfer between sensors
6. Store predictions as `HyImage` via `Stack.reshape()` for mapping/visualisation in hylite
