"""Tests for background transformations."""

from pathlib import Path

import cv2
import numpy as np

from synthocr.generate.transforms import Annotation, Sample, TextLine
from synthocr.generate.transforms.background import (
    HuggingFaceBackgroundTransform,
    RandomImageBackgroundTransform,
    ShadowLightingTransform,
)


def test_random_image_background_transform(tmp_path: Path) -> None:
    """Test random image background transformation."""
    # Create a fake background image
    bg_dir = tmp_path / "backgrounds"
    bg_dir.mkdir()
    bg_path = bg_dir / "bg.png"
    # Create a distinctive background (e.g., all blue)
    bg_img = np.zeros((200, 200, 3), dtype=np.uint8)
    bg_img[:, :, 2] = 255  # Blue in RGB
    cv2.imwrite(str(bg_path), cv2.cvtColor(bg_img, cv2.COLOR_RGB2BGR))

    # Create a sample with a transparent foreground (e.g., all red)
    # 4 channels (RGBA)
    fg_img = np.zeros((100, 100, 4), dtype=np.uint8)
    fg_img[:, :, 0] = 255  # Red
    fg_img[:, :, 3] = 255  # Fully opaque

    # But we want to test compositing, so let's make it half-transparent
    fg_img[50:, :, 3] = 0  # Bottom half fully transparent

    bbox = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    sample = Sample(
        image=fg_img,
        annotation=Annotation(lines=[TextLine(text="test", bbox=bbox)]),
    )

    transform = RandomImageBackgroundTransform(image_dir=str(bg_dir), seed=42)
    transformed_samples = transform([sample])
    transformed = transformed_samples[0]

    # Result should be RGB (3 channels)
    assert transformed.image.shape == (100, 100, 3)
    # Top half should be red (from foreground)
    assert np.all(transformed.image[0, 0] == [255, 0, 0])
    # Bottom half should be blue (from background)
    assert np.all(transformed.image[99, 99] == [0, 0, 255])
    assert len(transformed.annotation.lines) == 1


def test_huggingface_background_transform() -> None:
    """Test Hugging Face background transformation."""
    # Use a small dataset for testing, e.g., Fashion MNIST
    # Fashion MNIST has gray images, so it will test the mode conversion.
    transform = HuggingFaceBackgroundTransform(
        dataset_name="fashion_mnist", split="train", seed=42
    )

    # Create a 28x28 RGBA sample (same size as Fashion MNIST)
    fg_img = np.zeros((28, 28, 4), dtype=np.uint8)
    # Top half opaque white, bottom half transparent
    fg_img[:14, :, :3] = 255
    fg_img[:14, :, 3] = 255
    fg_img[14:, :, 3] = 0

    bbox = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    sample = Sample(
        image=fg_img,
        annotation=Annotation(lines=[TextLine(text="test", bbox=bbox)]),
    )

    transformed_samples = transform([sample])
    transformed = transformed_samples[0]

    # Result should be RGB
    assert transformed.image.shape == (28, 28, 3)
    # Top half should be white (from foreground)
    assert np.all(transformed.image[0, 0] == [255, 255, 255])
    # Bottom half should NOT be white (it should be some part of the fashion
    # MNIST image)
    # We can't easily predict the exact pixel, but it should be different
    assert not np.all(transformed.image[20, 20] == [255, 255, 255])


def test_shadow_lighting_transform() -> None:
    """Test shadow lighting transformation."""
    image = np.ones((200, 200, 4), dtype=np.uint8) * 200
    image[:, :, 3] = 255
    sample = Sample(image=image, annotation=Annotation(lines=[]))

    transform = ShadowLightingTransform(probability=1.0, seed=42)
    shadowed_samples = transform([sample])
    shadowed = shadowed_samples[0]

    # Shadowed image should be different
    assert not np.array_equal(shadowed.image, image)
