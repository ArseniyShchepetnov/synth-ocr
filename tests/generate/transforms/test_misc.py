"""Tests for miscellaneous transformations."""

import numpy as np

from synthocr.generate.transforms import Annotation, Sample, TextLine
from synthocr.generate.transforms.misc import CircleTransform, StampTransform


def test_stamp_transform() -> None:
    """Test stamp transformation."""
    image = np.ones((500, 500, 4), dtype=np.uint8) * 200
    image[:, :, 3] = 255
    sample = Sample(image=image, annotation=Annotation(lines=[]))

    transform = StampTransform(text="PAID APPROVED", probability=1.0, seed=42)
    stamped_samples = transform([sample])
    stamped = stamped_samples[0]

    # Stamped image should be different from original
    assert not np.array_equal(stamped.image, image)


def test_circle_transform() -> None:
    """Test circle transformation."""
    image = np.ones((100, 100, 4), dtype=np.uint8) * 255
    bbox = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    sample = Sample(
        image=image,
        annotation=Annotation(lines=[TextLine(text="test", bbox=bbox)]),
    )

    transform = CircleTransform(probability=1.0, seed=42)
    circled_samples = transform([sample])
    circled = circled_samples[0]

    # Circled image should be different
    assert not np.array_equal(circled.image, image)
