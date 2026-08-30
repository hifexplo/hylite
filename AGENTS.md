# Agent instructions

This repository ships an [Agent Skill](https://agentskills.io) for hylite at:

`.agents/skills/hylite/`

When the task involves hylite, hklearn, hycore, hywiz, hyperspectral images/clouds/libraries, drillcore processing, ENVI I/O, mineral mapping, or changes to this codebase, read `.agents/skills/hylite/SKILL.md` and follow it.

Do not assume the default `python` has hylite. Run `scripts/check_env.py` from that skill directory before executing code.

## Ecosystem

hylite is the core library. Related packages:

| Package | Role | Repository |
|---------|------|------------|
| **hylite** | Load, correct, project, and analyse hyperspectral data | https://github.com/hifexplo/hylite — docs: https://hifexplo.github.io/hylite/hylite.html |
| **hklearn** | Multi-sensor hyperspectral ML (`Stack`, `ModelSet`) | https://github.com/samthiele/hklearn |
| **hycore** | Drillcore data organisation and mosaics | https://github.com/samthiele/hycore |
| **hywiz** | Viewer for hycore hyperspectral core sheds | https://github.com/samthiele/hywiz |
| **ispec** | Interactive spectral libraries, mixing, feature ID | https://github.com/samthiele/ispec |
| **crunchy** | Multithreading for realtime processing workflows | https://github.com/samthiele/crunchy |
| **napari-hippo** | napari UI for some hylite tools | https://github.com/samthiele/napari-hippo |
| **speedy** | Browser-based hyperspectral image viewer (ENVI, overlays) | https://github.com/samthiele/speedy |
