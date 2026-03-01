"""Image transformations for OCR dataset generation.

Dynamic addition of transforms supports extension.
"""

import importlib
import inspect
from typing import Annotated, Any, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from synthocr.generate.transforms.base import (
    Annotation,
    BaseTransform,
    Sample,
    TextLine,
)
from synthocr.generate.transforms.settings import TRANSFORM_MODULES


def _discover_transforms() -> list[type[BaseTransform]]:
    """Discover all BaseTransform subclasses in specified modules."""
    transforms = []
    for module_name in TRANSFORM_MODULES:
        module = importlib.import_module(module_name, package=__package__)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseTransform)
                and obj is not BaseTransform
                and obj not in transforms
            ):
                transforms.append(obj)
    return transforms


ALL_TRANSFORMS = _discover_transforms()

# Export discovered transforms to the module namespace
for cls in ALL_TRANSFORMS:
    globals()[cls.__name__] = cls

# Define the dynamic discriminated union
TransformType = Annotated[
    Union[*ALL_TRANSFORMS],  # type: ignore[valid-type]
    Field(discriminator="name"),
]


class SampleTransformPipeline(BaseModel):
    """Sequential execution of image transformations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transforms: list[TransformType] = Field(default_factory=list)  # type: ignore[valid-type]

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply all transforms in the pipeline to a list of samples."""
        for transform in self.transforms:
            samples = transform(samples)
        return samples

    @classmethod
    def from_config(
        cls, config: list[dict[str, Any]]
    ) -> "SampleTransformPipeline":
        """Create a pipeline from a list of configuration dictionaries."""
        return cls.model_validate({"transforms": config})


def get_pipeline_schema() -> dict[str, Any]:
    """Generate a JSON schema for the SampleTransformPipeline."""
    return SampleTransformPipeline.model_json_schema()


__all__ = [
    "TextLine",
    "Annotation",
    "Sample",
    "SampleTransformPipeline",
    "TransformType",
    "BaseTransform",
    "get_pipeline_schema",
] + [cls.__name__ for cls in ALL_TRANSFORMS]
