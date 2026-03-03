"""Tests for distortion transformations."""

import numpy as np

from synthocr.generate.transforms import Annotation, Sample, TextLine
from synthocr.generate.transforms.distortion import (
    CylindricalTransform,
    RotationTransform,
    WaveTransform,
)


def test_rotation_transform() -> None:
    """Test rotation transformation."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    bbox = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    sample = Sample(
        image=image,
        annotation=Annotation(lines=[TextLine(text="test", bbox=bbox)]),
    )

    transform = RotationTransform(max_angle=10, seed=42)
    rotated_samples = transform([sample])
    rotated = rotated_samples[0]

    assert rotated.image.shape == sample.image.shape
    assert len(rotated.annotation.lines) == 1
    assert not np.array_equal(rotated.annotation.lines[0].bbox, bbox)


def test_cylindrical_transform() -> None:
    """Test cylindrical transformation."""
    image = np.ones((100, 100, 4), dtype=np.uint8) * 255
    bbox = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    sample = Sample(
        image=image,
        annotation=Annotation(lines=[TextLine(text="test", bbox=bbox)]),
    )

    transform = CylindricalTransform(curvature=0.01, seed=42, probability=1.0)
    transformed_samples = transform([sample])
    transformed = transformed_samples[0]

    assert transformed.image.shape == sample.image.shape
    assert len(transformed.annotation.lines) == 1
    assert not np.array_equal(transformed.annotation.lines[0].bbox, bbox)


def test_wave_transform() -> None:
    """Test wave transformation."""
    image = np.ones((100, 100, 4), dtype=np.uint8) * 255
    bbox = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    sample = Sample(
        image=image,
        annotation=Annotation(lines=[TextLine(text="test", bbox=bbox)]),
    )

    transform = WaveTransform(
        amplitude=5.0, period=50.0, direction="vertical", seed=42
    )
    transformed_samples = transform([sample])
    transformed = transformed_samples[0]

    assert transformed.image.shape == sample.image.shape
    assert len(transformed.annotation.lines) == 1
    assert not np.array_equal(transformed.annotation.lines[0].bbox, bbox)
