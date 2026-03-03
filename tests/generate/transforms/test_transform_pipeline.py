"""Tests for the transformation pipeline."""

from typing import Any

from synthocr.generate.transforms import SampleTransformPipeline
from synthocr.generate.transforms.background import (
    HuggingFaceBackgroundTransform,
    RandomImageBackgroundTransform,
)
from synthocr.generate.transforms.distortion import RotationTransform
from synthocr.generate.transforms.misc import GaussianNoiseTransform


def test_pipeline_from_config() -> None:
    """Test pipeline creation from configuration dictionary."""
    config: list[dict[str, Any]] = [
        {"name": "RotationTransform", "max_angle": 10},
        {"name": "GaussianNoiseTransform", "sigma": 5.0},
        {
            "name": "RandomImageBackgroundTransform",
            "image_dir": "test_dir",
        },
        {
            "name": "HuggingFaceBackgroundTransform",
            "dataset_name": "fashion_mnist",
        },
    ]
    pipeline = SampleTransformPipeline.from_config(config)
    assert len(pipeline.transforms) == 4
    assert isinstance(pipeline.transforms[0], RotationTransform)
    assert isinstance(pipeline.transforms[1], GaussianNoiseTransform)
    assert isinstance(pipeline.transforms[2], RandomImageBackgroundTransform)
    assert isinstance(pipeline.transforms[3], HuggingFaceBackgroundTransform)


def test_pipeline_from_config_snake_case() -> None:
    """Test pipeline creation from config with snake_case names."""
    config: list[dict[str, Any]] = [
        {"name": "rotation_transform", "max_angle": 10},
        {"name": "gaussian_noise_transform", "sigma": 5.0},
    ]
    pipeline = SampleTransformPipeline.from_config(config)
    assert len(pipeline.transforms) == 2
    assert isinstance(pipeline.transforms[0], RotationTransform)
    assert isinstance(pipeline.transforms[1], GaussianNoiseTransform)
    assert pipeline.transforms[0].name in (
        "RotationTransform",
        "rotation_transform",
    )
    assert pipeline.transforms[1].name in (
        "GaussianNoiseTransform",
        "gaussian_noise_transform",
    )
