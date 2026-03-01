"""Base classes and utilities for image transformations."""

import abc
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel


@dataclass
class TextLine:
    """Represents a single line of text with its polygon bounding box."""

    text: str
    bbox: np.ndarray


@dataclass
class Annotation:
    """Contains a collection of text lines for an image."""

    lines: list[TextLine]


@dataclass
class Sample:
    """Container for an image and its corresponding annotation."""

    image: np.ndarray
    annotation: Annotation


class BaseTransform(BaseModel, abc.ABC):
    """Abstract base class for all image transformations."""

    name: str

    @abc.abstractmethod
    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the transformation to a list of samples."""
        ...


def sample_param(param: Any, rng: random.Random) -> Any:  # noqa: ANN401
    """Return a random value within a range or the parameter itself."""
    range_len = 2
    if isinstance(param, (list, tuple)) and len(param) == range_len:
        return rng.uniform(param[0], param[1])
    return param


def sample_int_param(param: Any, rng: random.Random) -> Any:  # noqa: ANN401
    """Return a random integer within a range or the parameter itself."""
    range_len = 2
    if isinstance(param, (list, tuple)) and len(param) == range_len:
        return rng.randint(int(param[0]), int(param[1]))
    return param
