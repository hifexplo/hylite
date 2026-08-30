"""Shared helpers for tier-aware hylite tests."""

import os
from pathlib import Path

from hylite._deps import simulateEnv

TEST_DATA = os.path.join(str(Path(__file__).parent.parent), "test_data")


def require_test_env(testcase, tier):
    """
    Simulate *tier* and skip *testcase* when the environment lacks its packages.

    *tier* is one of: ``basic``, ``lite``, ``default``, ``opencv``, ``gdal``, ``all``.

    Call at the start of each test method, e.g.::

        require_test_env(self, "default")

    For tests that walk up through tiers in one method, start with
    :func:`require_test_env` at the lowest tier and then::

        if not upgrade_test_env("lite"):
            return
    """
    if not simulateEnv(tier):
        testcase.skipTest(
            "requires hylite[%s]: not all packages for this tier are installed"
            % tier
        )


def upgrade_test_env(tier):
    """
    Advance the simulated install target part-way through a test.

    Returns ``True`` if every package for *tier* is installed (continue testing at
    this level). Returns ``False`` if packages are missing (stop upgrading; lower
    tiers have already run).
    """
    return simulateEnv(tier)
