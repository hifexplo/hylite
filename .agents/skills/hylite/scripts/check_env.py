#!/usr/bin/env python3
"""Probe that hylite is importable. Exit 0 on success, 1 with MISSING_DEPS."""
import sys

missing = []

try:
    import hylite 
except ImportError:
    missing.append("hylite")

if missing:
    print("MISSING_DEPS: " + ", ".join(missing))
    sys.exit(1)

print("SUCCESS: hylite is available.")
print("PYTHON:", sys.executable)
