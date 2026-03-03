"""Tests for dynamic transform discovery."""

import sys
import types
from typing import Literal

from synthocr.generate.transforms.base import (
    BaseTransform,
    Sample,
    discover_transforms,
)


def test_discover_transforms_external() -> None:
    """Test discovering transforms from an external (mocked) module."""
    # Create a mock module
    module_name = "mock_transform_module"
    mock_module = types.ModuleType(module_name)
    sys.modules[module_name] = mock_module

    # Define a new transform in the mock module
    class MockTransform(BaseTransform):
        name: Literal["mock_transform"] = "mock_transform"

        def __call__(self, samples: list[Sample]) -> list[Sample]:
            return samples

    mock_module.MockTransform = MockTransform  # type: ignore[attr-defined]

    try:
        # Discover transforms including the mock module
        transforms = discover_transforms(modules=[], external=[module_name])

        assert MockTransform in transforms
        assert len(transforms) == 1
        assert transforms[0].model_fields["name"].default == "mock_transform"

    finally:
        # Clean up sys.modules
        if module_name in sys.modules:
            del sys.modules[module_name]


def test_discover_transforms_empty() -> None:
    """Test discovery with no modules."""
    transforms = discover_transforms(modules=[])
    assert len(transforms) == 0


def test_discover_transforms_invalid_module() -> None:
    """Test discovery with an invalid module name (should not crash)."""
    transforms = discover_transforms(modules=["non_existent_module_xyz"])
    assert len(transforms) == 0
