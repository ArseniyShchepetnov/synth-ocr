"""Tests for mixing transformations."""

import numpy as np

from synthocr.generate.transforms import Annotation, Sample, TextLine
from synthocr.generate.transforms.mix import MixBatchTransform


def test_mix_batch_transform() -> None:
    """Test batch mixing transformation."""
    # Create two distinct samples
    img1 = np.zeros((100, 100, 4), dtype=np.uint8)
    img1[:, :, 0] = 255  # Red
    img1[:, :, 3] = 255
    bbox1 = np.array(
        [[10, 10], [40, 10], [40, 20], [10, 20]], dtype=np.float32
    )
    sample1 = Sample(
        image=img1,
        annotation=Annotation(lines=[TextLine(text="red", bbox=bbox1)]),
    )

    img2 = np.zeros((100, 100, 4), dtype=np.uint8)
    img2[:, :, 1] = 255  # Green
    img2[:, :, 3] = 255
    bbox2 = np.array(
        [[50, 50], [80, 50], [80, 60], [50, 60]], dtype=np.float32
    )
    sample2 = Sample(
        image=img2,
        annotation=Annotation(lines=[TextLine(text="green", bbox=bbox2)]),
    )

    transform = MixBatchTransform(m=2, k=1, min_lines=2, max_lines=2, seed=42)
    mixed_samples = transform([sample1, sample2])

    assert len(mixed_samples) == 1
    mixed = mixed_samples[0]
    assert mixed.image.shape == (100, 100, 4)
    assert len(mixed.annotation.lines) == 2
    texts = [line.text for line in mixed.annotation.lines]
    assert "red" in texts
    assert "green" in texts
