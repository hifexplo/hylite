# Extend or edit the hylite codebase

**Goal:** Change hylite source, tests, or API — not analyse a specific dataset.

```
- [ ] Environment probe (`scripts/check_env.py`)
- [ ] Identify scope: bug fix, new type, test, docs
- [ ] Minimal diff; match existing conventions
- [ ] Run targeted tests
```

## Conventions

- Follow patterns in neighbouring modules; reuse existing helpers.
- Keep code clear and minimal. Avoid private helpers unless used at least twice.
- One short explanatory comment per block of code.
- Each main function needs a docstring matching repository style.
- Run relevant tests after edits. Use the same `PYTHON:` from the environment probe.

## Adding features

- New sensors: subclass `hylite.sensors.Sensor`, add calibration data under `sensors/calibration_data/`.
- New I/O: extend `io.load` / `io.save` dispatch in `hylite/io/__init__.py`.
- Wavelength-aware band access via `[]` or `get_band_index()`.
- Do not add extra functions (including private `_` helpers) unless they are reused.

For API behaviour, read docstrings and [reference.md](reference.md). Verify against tests before documenting new behaviour in this skill.
