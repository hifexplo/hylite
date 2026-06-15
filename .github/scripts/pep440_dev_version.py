#!/usr/bin/env python3
"""Set setup.py version to a PEP 440 dev release for CI dev-branch wheels."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def dev_version(version: str) -> str:
    if re.search(r"\.dev\d", version):
        return version
    return f"{version}.dev0"


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: pep440_dev_version.py <setup.py>", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', text)
    if not match:
        print(f"no version= found in {path}", file=sys.stderr)
        sys.exit(1)

    new_version = dev_version(match.group(1))
    text = re.sub(
        r'(version\s*=\s*")[^"]+(")',
        rf"\g<1>{new_version}\2",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    print(new_version)


if __name__ == "__main__":
    main()
