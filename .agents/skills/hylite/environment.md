# Python environment for hylite

Run this protocol **before** using hylite in scripts,tests or notebooks. Do **not** assume the IDE’s default `python` has `hylite`.

Use one **verified interpreter** for the whole session (record `PYTHON:` from the probe). Environments may be named anything (`hylite`, `hsi`, `hifsi`, `base`, …) and may come from **mamba**, **conda**, **micromamba**, **venv**, **uv**, **Poetry**, **pipenv**, or a plain path to `bin/python`.

**Quick probe** (from this skill directory):

```bash
python scripts/check_env.py
```

---

## Step 0: Pick how to invoke Python (optional)

If the user already said which env to use, skip discovery. Otherwise probe in this order:

| Situation | How to run the probe |
|-----------|----------------------|
| Shell already activated | `python scripts/check_env.py` |
| Conda-style env **name** `ENV` | `mamba run -n ENV python scripts/check_env.py` **or** `conda run -n ENV python …` **or** `micromamba run -n ENV python …` (use whichever command exists: `command -v mamba`, `conda`, `micromamba`) |
| **Path** to interpreter | `"<path/to/python>" scripts/check_env.py` |
| Project **venv** | `.venv/bin/python scripts/check_env.py` (or `source .venv/bin/activate` then `python scripts/check_env.py`) |
| **uv** | `uv run python scripts/check_env.py` if the project uses uv |
| **Poetry** | `poetry run python scripts/check_env.py` |

Do not hardcode the env name `hylite`. For this repo, a common setup is `mamba activate hylite` — treat that as an example, not a requirement.

---

## Step 1: Probe the runtime

Do not guess. Run the probe with the interpreter you will use for the task.

**Probe script** — run `python scripts/check_env.py` or inline equivalent (only the `python` prefix changes):

```bash
python scripts/check_env.py
```

Examples (replace `ENV` / paths as needed):

```bash
# Active env
python scripts/check_env.py

# Named conda/mamba/micromamba env (pick the tool that exists)
mamba run -n ENV python scripts/check_env.py
conda run -n ENV python scripts/check_env.py
micromamba run -n ENV python scripts/check_env.py

# Explicit interpreter
/path/to/envs/ENV/bin/python scripts/check_env.py

# Local venv
.venv/bin/python scripts/check_env.py
```

After success, store `PYTHON:` and use **that binary** (or the same activated env) for all later commands.

---

## Step 2: Interpret the result

| Result | Action |
|--------|--------|
| Exit code **0** | Confirm success with the printed `PYTHON:` path. Proceed with hylite using that interpreter. |
| Exit code **1** (`MISSING_DEPS: …`) | **Stop.** Do not run training, fitting, or `pytest`. Go to Step 3. |
| Wrong / missing `python` | Ask the user for an env name, activation command, or full path to `python`. |

Do **not** `pip install` into an unknown or system interpreter without user consent.

---

## Step 3: Ask the user (required if probe failed)

Halt autonomous execution and ask using this format:

> **Environment configuration required**
>
> `[missing packages]` is not available in the current Python runtime (`[python path probed, if known]`).
>
> To continue, reply with one of:
>
> - **Activation or env id** — e.g. `mamba activate my-env`, `conda activate geo`, `source .venv/bin/activate`, `poetry shell`, or the env name alone (`my-env`) if conda/mamba/micromamba is available.
> - **Path to `python`** — e.g. `/Users/…/miniconda3/envs/my-env/bin/python`, `.venv/bin/python`.
> - **`install`** — allow creating a local environment here (Step 4B).
> - **`global`** — install into the **currently probed** interpreter (only if the user explicitly accepts the risk).

If the user has not specified an env, ask once; do not assume a particular env name or package manager.

---

## Step 4: Route the user’s reply

### Case A: Named env, activation line, or path

1. **Full path to `python`:** use `"<path>" scripts/check_env.py`, `"<path>" -m pytest`, `"<path>" script.py` for the rest of the session.
2. **Env name only** (e.g. `hsi`, `hylite`): run probes and scripts via `mamba run -n <name> …`, or `conda run -n <name> …`, or `micromamba run -n <name> …` — prefer the tool the user mentioned or that exists on `PATH`.
3. **Activation command** (e.g. `conda activate geo`, `source .venv/bin/activate`): run that in the shell, then use `python` from that shell for Step 1 and all follow-up work.
4. **Poetry / uv / pipenv:** use `poetry run`, `uv run`, or `pipenv run` consistently if the user indicated that workflow.
5. Re-run **Step 1**; only continue on `SUCCESS`.

### Case B: User typed `install`

Prefer what the user’s machine already has; otherwise use a **local venv** in the project (portable, no admin):

**Option 1 — conda family** (if `mamba`, `conda`, or `micromamba` is available):

```bash
# Example; user may choose ENV name
mamba create -n ENV python=3.11 -y   # or: conda create -n ENV ...
mamba activate ENV                   # or: conda activate ENV
pip install --upgrade pip
pip install torch
```

**Option 2 — venv** (always valid if `python3` exists):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch
```

Then install hylite:

- **Inside this git clone:** `pip install -e .`
- **Elsewhere:** `pip install "hylite"` or `pip install git+https://github.com/hifexplo/hylite.git`

Re-run Step 1. Report `PYTHON:` to the user.

### Case C: User typed `global`

Warn that this affects the interpreter that was probed. Install only into **that** `python`:

```bash
python -m pip install --upgrade pip
python -m pip install "hylite"   # or -e ".[all]" in the repo clone
```

Re-run Step 1.

---

## Step 5: Session rules after success

- Reuse the **same** `PYTHON:` (or the same activated env / `run -n` wrapper) for `pytest`, sandbox scripts, and notebooks.
- In a **new shell**, re-activate or use `mamba run -n ENV` / `conda run -n ENV` / `.venv/bin/python` — do not fall back to an unprobed system `python`.
- Developing on the **dev** branch in this repo: `pip install -e ".[all]"` in the verified env so imports match local code.