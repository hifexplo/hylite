# Extend or edit the hylite codebase

**Goal:** Change hylite source, tests, or API — not analyse a specific dataset.

## Workflow

```
- [ ] Environment probe (scripts/check_env.py)
- [ ] Identify scope: bug fix, new field type, synthetic builder, test, docs
- [ ] Minimal diff; match existing conventions
- [ ] Run targeted tests
```

## Conventions

- Follow patterns in neighbouring modules; reuse existing helpers.
- Keep code clear and minimal. Avoid writing private functions where possible. Helper functions must be used at least twice, otherwise they should be integrated with the function using them.
- Provide one short explanatory comment per block of code (at least)
- Ensure each main function has a docstring, matching the style of the overall repository
- Ensure relevant tests are run after making edits.

Use the same `PYTHON:` from the environment probe for all commands.

For API behaviour, read docstrings and [reference.md](../reference.md); verify against tests before documenting new behaviour in skills.
