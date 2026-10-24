"""Scaffolding smoke test (Task 1.1).

Confirms every pipeline package named in the design's layout-mapping table
is importable. This guards the shared foundation that all later tasks build
on; module-specific tests live under the matching ``tests/<package>/``
subfolder.
"""

import importlib

import pytest

PIPELINE_PACKAGES = ["data", "flag", "eval", "app", "export", "output"]


@pytest.mark.parametrize("package", PIPELINE_PACKAGES)
def test_pipeline_package_is_importable(package):
    module = importlib.import_module(package)
    assert module is not None
