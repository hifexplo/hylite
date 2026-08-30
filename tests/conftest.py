"""Pytest hooks for hylite tier simulation."""

from hylite._deps import resetSimulateEnv


def pytest_runtest_teardown(item, nextitem):
    resetSimulateEnv()
