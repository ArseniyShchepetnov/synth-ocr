"""Background transformations for OCR dataset generation."""

import random
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from datasets import load_dataset
from pydantic import ConfigDict, Field

from synthocr.generate.transforms.base import BaseTransform, Sample


class BackgroundFillTransform(BaseTransform):
    """Fills the transparent background with a solid color.

    The algorithm uses alpha blending to composite the existing image over
    a solid background color. If no color is provided, a random bright
    color is generated.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["background_fill_transform", "BackgroundFillTransform"] = (
        "BackgroundFillTransform"
    )
    color: tuple[int, int, int] | list[int] | None = None
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the background fill transformation."""
        output = []
        rgba_channels = 4
        rgb_channels = 3
        for sample in samples:
            if sample.image.shape[2] != rgba_channels:
                output.append(sample)
                continue

            if self.rng.random() > self.probability:
                output.append(
                    Sample(
                        image=sample.image[:, :, :rgb_channels],
                        annotation=sample.annotation,
                    )
                )
                continue

            h, w = sample.image.shape[:2]

            if (
                self.color
                and isinstance(self.color, (list, tuple))
                and len(self.color) == rgb_channels
            ):
                bg_color = self.color
            else:
                bg_color = (
                    self.rng.randint(200, 255),
                    self.rng.randint(200, 255),
                    self.rng.randint(200, 255),
                )

            alpha = (sample.image[:, :, 3] / 255.0)[:, :, np.newaxis]
            rgb = sample.image[:, :, :rgb_channels]
            bg = np.full((h, w, 3), bg_color, dtype=np.uint8)

            final_rgb = (alpha * rgb + (1 - alpha) * bg).astype(np.uint8)
            output.append(
                Sample(image=final_rgb, annotation=sample.annotation)
            )

        return output


class TextureBackgroundTransform(BaseTransform):
    """Adds a procedural paper texture and optional creases.

    The algorithm generates a "paper grain" using Gaussian noise on a solid
    canvas, adds linear crease artifacts, and composites the original sample
    over this background using alpha blending.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal[
        "texture_background_transform", "TextureBackgroundTransform"
    ] = "TextureBackgroundTransform"
    textures_dir: str | None = None
    crease_probability: float = 0.5
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the texture background transformation."""
        output = []
        rgba_channels = 4
        for sample in samples:
            if (
                self.rng.random() > self.probability
                or sample.image.shape[2] != rgba_channels
            ):
                output.append(sample)
                continue

            h, w = sample.image.shape[:2]

            bg = np.full((h, w, 3), 245, dtype=np.uint8)
            grain = np.random.normal(0, 5, (h, w, 3)).astype(np.int16)
            bg = np.clip(bg.astype(np.int16) + grain, 0, 255).astype(np.uint8)

            if self.rng.random() < self.crease_probability:
                for _ in range(self.rng.randint(1, 4)):
                    p1 = (self.rng.randint(0, w), 0)
                    p2 = (self.rng.randint(0, w), h)
                    cv2.line(bg, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

            alpha = (sample.image[:, :, 3] / 255.0)[:, :, np.newaxis]
            rgb = sample.image[:, :, :3]

            final_rgb = (alpha * rgb + (1 - alpha) * bg).astype(np.uint8)
            output.append(
                Sample(image=final_rgb, annotation=sample.annotation)
            )
        return output


class RandomImageBackgroundTransform(BaseTransform):
    """Adds a random background image from a directory.

    The algorithm selects a random image from the specified directory,
    resizes and crops it to match the sample dimensions, and alpha-composites
    the sample onto it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal[
        "random_image_background_transform", "RandomImageBackgroundTransform"
    ] = "RandomImageBackgroundTransform"
    image_dir: str
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    @cached_property
    def _image_paths(self) -> list[Path]:
        """List of image paths found in the image_dir."""
        image_dir_path = Path(self.image_dir)
        if image_dir_path.exists():
            return [
                f
                for f in image_dir_path.iterdir()
                if f.suffix.lower()
                in (".png", ".jpg", ".jpeg", ".bmp", ".webp")
            ]
        return []

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the random image background transformation."""
        if not self._image_paths:
            return samples

        output: list[Sample] = []
        rgba_channels = 4
        for sample in samples:
            if (
                self.rng.random() > self.probability
                or sample.image.shape[2] != rgba_channels
            ):
                output.append(sample)
                continue

            h, w = sample.image.shape[:2]
            bg_path = self.rng.choice(self._image_paths)
            bg_img_bgr = cv2.imread(str(bg_path))

            if bg_img_bgr is None:
                output.append(sample)
                continue

            bg_img = cv2.cvtColor(bg_img_bgr, cv2.COLOR_BGR2RGB)

            bg_h, bg_w = bg_img.shape[:2]
            if bg_h < h or bg_w < w:
                scale = max(h / bg_h, w / bg_w)
                new_h, new_w = int(bg_h * scale) + 1, int(bg_w * scale) + 1
                bg_img = cv2.resize(
                    bg_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                bg_h, bg_w = bg_img.shape[:2]

            top = self.rng.randint(0, bg_h - h)
            left = self.rng.randint(0, bg_w - w)
            bg_crop = bg_img[top : top + h, left : left + w]

            alpha = (sample.image[:, :, 3] / 255.0)[:, :, np.newaxis]
            rgb = sample.image[:, :, :3]

            final_rgb = (alpha * rgb + (1.0 - alpha) * bg_crop).astype(
                np.uint8
            )
            output.append(
                Sample(image=final_rgb, annotation=sample.annotation)
            )

        return output


class HuggingFaceBackgroundTransform(BaseTransform):
    """Adds a random background image from a Hugging Face dataset.

    The algorithm loads a dataset from Hugging Face, selects a random
    image from it, resizes and crops it to match the sample dimensions,
    and alpha-composites the sample onto it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal[
        "hugging_face_background_transform", "HuggingFaceBackgroundTransform"
    ] = "HuggingFaceBackgroundTransform"
    dataset_name: str
    split: str = "train"
    image_column: str = "image"
    probability: float = 1.0
    seed: int | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    @cached_property
    def dataset(self) -> Any:  # noqa: ANN401
        """Loaded Hugging Face dataset."""
        return load_dataset(self.dataset_name, split=self.split, **self.kwargs)

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the Hugging Face background transformation."""
        output: list[Sample] = []
        rgba_channels = 4
        for sample in samples:
            if (
                self.rng.random() > self.probability
                or sample.image.shape[2] != rgba_channels
            ):
                output.append(sample)
                continue

            h, w = sample.image.shape[:2]

            # Pick a random index from the dataset
            idx = self.rng.randint(0, len(self.dataset) - 1)
            item = self.dataset[idx]
            bg_pil = item[self.image_column]

            # Ensure it's an RGB image
            if bg_pil.mode != "RGB":
                bg_pil = bg_pil.convert("RGB")

            bg_img = np.array(bg_pil)

            bg_h, bg_w = bg_img.shape[:2]
            if bg_h < h or bg_w < w:
                scale = max(h / bg_h, w / bg_w)
                new_h, new_w = int(bg_h * scale) + 1, int(bg_w * scale) + 1
                bg_img = cv2.resize(
                    bg_img, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                bg_h, bg_w = bg_img.shape[:2]

            top = self.rng.randint(0, bg_h - h)
            left = self.rng.randint(0, bg_w - w)
            bg_crop = bg_img[top : top + h, left : left + w]

            alpha = (sample.image[:, :, 3] / 255.0)[:, :, np.newaxis]
            rgb = sample.image[:, :, :3]

            final_rgb = (alpha * rgb + (1.0 - alpha) * bg_crop).astype(
                np.uint8
            )
            output.append(
                Sample(image=final_rgb, annotation=sample.annotation)
            )

        return output


class ShadowLightingTransform(BaseTransform):
    """Simulates uneven lighting or shadows.

    The algorithm creates a linear gradient mask and multiplies it with
    the RGB channels of the image to simulate a light source coming from
    one of the edges.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["shadow_lighting_transform", "ShadowLightingTransform"] = (
        "ShadowLightingTransform"
    )
    probability: float = 0.6
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the shadow lighting transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            h, w = sample.image.shape[:2]
            mask = np.ones((h, w), dtype=np.float32)

            if self.rng.choice([True, False]):
                grad = np.linspace(self.rng.uniform(0.4, 0.8), 1.0, h).astype(
                    np.float32
                )
                mask = mask * grad[:, np.newaxis]
            else:
                grad = np.linspace(self.rng.uniform(0.4, 0.8), 1.0, w).astype(
                    np.float32
                )
                mask = mask * grad[np.newaxis, :]

            img = sample.image.copy()
            for c in range(3):
                img[:, :, c] = (img[:, :, c].astype(np.float32) * mask).astype(
                    np.uint8
                )

            output.append(Sample(image=img, annotation=sample.annotation))
        return output
