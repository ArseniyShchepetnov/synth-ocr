"""Miscellaneous image transformations for OCR dataset generation."""

import random
from functools import cached_property
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import ConfigDict, Field

from synthocr.generate.transforms.base import (
    BaseTransform,
    Sample,
    sample_int_param,
    sample_param,
)


class GaussianNoiseTransform(BaseTransform):
    """Adds random Gaussian noise to the image.

    The algorithm generates noise from a normal distribution and adds it
    exclusively to the RGB channels, preserving the alpha channel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["gaussian_noise_transform", "GaussianNoiseTransform"] = (
        "GaussianNoiseTransform"
    )
    sigma: float | list[float]
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the Gaussian noise transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            sigma = sample_param(self.sigma, self.rng)

            img_rgb = sample.image[:, :, :3].astype(np.float32)
            noise = np.random.normal(0, sigma, img_rgb.shape).astype(
                np.float32
            )
            noisy_rgb = np.clip(img_rgb + noise, 0, 255).astype(np.uint8)

            noisy_image = np.copy(sample.image)
            noisy_image[:, :, :3] = noisy_rgb

            output.append(
                Sample(image=noisy_image, annotation=sample.annotation)
            )
        return output


class DilationTransform(BaseTransform):
    """Applies morphological dilation to the image.

    The algorithm uses a square kernel of a sampled size to expand
    the brighter regions in the image.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["dilation_transform", "DilationTransform"] = (
        "DilationTransform"
    )
    kernel_size: int | list[int] = 3
    iterations: int | list[int] = 1
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the dilation transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            k_size = sample_int_param(self.kernel_size, self.rng)
            if k_size % 2 == 0:
                k_size += 1
            iters = sample_int_param(self.iterations, self.rng)

            kernel = np.ones((k_size, k_size), np.uint8)
            dilated_image = cv2.dilate(sample.image, kernel, iterations=iters)
            output.append(
                Sample(image=dilated_image, annotation=sample.annotation)
            )
        return output


class ErosionTransform(BaseTransform):
    """Applies morphological erosion to the image.

    The algorithm uses a square kernel of a sampled size to shrink
    the brighter regions in the image.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["erosion_transform", "ErosionTransform"] = "ErosionTransform"
    kernel_size: int | list[int] = 3
    iterations: int | list[int] = 1
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the erosion transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            k_size = sample_int_param(self.kernel_size, self.rng)
            if k_size % 2 == 0:
                k_size += 1
            iters = sample_int_param(self.iterations, self.rng)

            kernel = np.ones((k_size, k_size), np.uint8)
            eroded_image = cv2.erode(sample.image, kernel, iterations=iters)
            output.append(
                Sample(image=eroded_image, annotation=sample.annotation)
            )
        return output


class BrightnessContrastTransform(BaseTransform):
    """Adjusts the brightness and contrast of the image.

    The algorithm uses linear scaling applied only to the RGB channels
    while maintaining the original alpha channel.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal[
        "brightness_contrast_transform", "BrightnessContrastTransform"
    ] = "BrightnessContrastTransform"
    brightness: float | list[float] = 0.0
    contrast: float | list[float] = 1.0
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the brightness and contrast transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            bright = sample_param(self.brightness, self.rng)
            cont = sample_param(self.contrast, self.rng)

            rgb = sample.image[:, :, :3]
            alpha = sample.image[:, :, 3:]
            new_rgb = cv2.convertScaleAbs(rgb, alpha=cont, beta=bright)
            new_image = np.concatenate([new_rgb, alpha], axis=2)
            output.append(
                Sample(image=new_image, annotation=sample.annotation)
            )
        return output


class MotionBlurTransform(BaseTransform):
    """Applies a horizontal or vertical motion blur.

    The algorithm creates a 1D kernel oriented randomly and applies it
    using a 2D filter to simulate movement blur.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["motion_blur_transform", "MotionBlurTransform"] = (
        "MotionBlurTransform"
    )
    kernel_size: int | list[int] = 5
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the motion blur transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            k_size = sample_int_param(self.kernel_size, self.rng)
            if k_size % 2 == 0:
                k_size += 1

            kernel = np.zeros((k_size, k_size))
            if self.rng.choice([True, False]):
                kernel[int((k_size - 1) / 2), :] = np.ones(k_size)
            else:
                kernel[:, int((k_size - 1) / 2)] = np.ones(k_size)
            kernel /= k_size
            blurred_image = cv2.filter2D(sample.image, -1, kernel)
            output.append(
                Sample(image=blurred_image, annotation=sample.annotation)
            )
        return output


class SaltAndPepperNoiseTransform(BaseTransform):
    """Adds salt and pepper (impulse) noise.

    The algorithm randomly sets pixels to either full white or full black
    based on the specified amount and distribution ratio.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal[
        "salt_and_pepper_noise_transform", "SaltAndPepperNoiseTransform"
    ] = "SaltAndPepperNoiseTransform"
    amount: float | list[float] = 0.01
    salt_vs_pepper: float | list[float] = 0.5
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the salt and pepper noise transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            amt = sample_param(self.amount, self.rng)
            svp = sample_param(self.salt_vs_pepper, self.rng)

            out = np.copy(sample.image)
            num_salt = np.ceil(amt * sample.image.size * svp)
            coords = [
                self.rng.randint(0, i - 1)
                for i in sample.image.shape
                for _ in range(int(num_salt))
            ]
            # Reshape coords for advanced indexing
            coords_tuple = tuple(
                np.array(coords).reshape(len(sample.image.shape), -1)
            )
            out[coords_tuple] = 255

            num_pepper = np.ceil(amt * sample.image.size * (1.0 - svp))
            coords_p = [
                self.rng.randint(0, i - 1)
                for i in sample.image.shape
                for _ in range(int(num_pepper))
            ]
            coords_p_tuple = tuple(
                np.array(coords_p).reshape(len(sample.image.shape), -1)
            )
            out[coords_p_tuple] = 0

            output.append(Sample(image=out, annotation=sample.annotation))
        return output


class StampTransform(BaseTransform):
    """Applies a realistic rubber stamp effect.

    The algorithm creates a separate RGBA buffer with text and a circular
    border, applies artifacts like Gaussian blur and random pixel dropout,
    rotates it, and alpha-composites it onto the image.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["stamp_transform", "StampTransform"] = "StampTransform"
    text: str = "PAID"
    font_path: str | None = None
    probability: float = 0.5
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the rubber stamp transformation."""
        output = []
        rgba_channels = 4
        dropout_threshold = 30
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            h, w = sample.image.shape[:2]
            stamp_size = 200
            stamp_img = Image.new(
                "RGBA", (stamp_size, stamp_size), (0, 0, 0, 0)
            )
            draw = ImageDraw.Draw(stamp_img)

            ink_color = (
                self.rng.randint(150, 220),
                self.rng.randint(20, 60),
                self.rng.randint(20, 100),
                180,
            )

            font: ImageFont.FreeTypeFont | ImageFont.ImageFont
            try:
                if self.font_path:
                    font = ImageFont.truetype(self.font_path, 40)
                else:
                    font = ImageFont.load_default()
            except OSError:
                font = ImageFont.load_default()

            draw.ellipse([10, 10, 190, 190], outline=ink_color, width=5)
            tw, th = draw.textbbox((0, 0), self.text, font=font)[2:4]
            draw.text(
                ((stamp_size - tw) / 2, (stamp_size - th) / 2),
                self.text,
                font=font,
                fill=ink_color,
            )

            stamp_arr = np.array(stamp_img)
            stamp_arr = cv2.GaussianBlur(stamp_arr, (3, 3), 0)

            mask = stamp_arr[:, :, 3] > 0
            noise = np.array(
                [
                    [self.rng.randint(0, 255) for _ in range(stamp_size)]
                    for _ in range(stamp_size)
                ],
                dtype=np.uint8,
            )
            stamp_arr[mask & (noise < dropout_threshold), 3] = 0

            angle = self.rng.uniform(-30, 30)
            rot_matrix = cv2.getRotationMatrix2D(
                (stamp_size / 2, stamp_size / 2), angle, 1.0
            )
            stamp_arr = cv2.warpAffine(
                stamp_arr,
                rot_matrix,
                (stamp_size, stamp_size),
                borderValue=(0, 0, 0, 0),
            )

            tx = self.rng.randint(0, max(1, w - stamp_size))
            ty = self.rng.randint(0, max(1, h - stamp_size))

            image_rgba = (
                sample.image
                if sample.image.shape[2] == rgba_channels
                else cv2.cvtColor(sample.image, cv2.COLOR_RGB2RGBA)
            )

            actual_stamp_h = min(stamp_size, h - ty)
            actual_stamp_w = min(stamp_size, w - tx)

            stamp_crop = stamp_arr[:actual_stamp_h, :actual_stamp_w]
            alpha_s = stamp_crop[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(3):
                image_rgba[
                    ty : ty + actual_stamp_h, tx : tx + actual_stamp_w, c
                ] = (
                    alpha_s * stamp_crop[:, :, c]
                    + alpha_l
                    * image_rgba[
                        ty : ty + actual_stamp_h, tx : tx + actual_stamp_w, c
                    ]
                ).astype(np.uint8)

            if sample.image.shape[2] == rgba_channels:
                image_rgba[
                    ty : ty + actual_stamp_h, tx : tx + actual_stamp_w, 3
                ] = np.maximum(
                    image_rgba[
                        ty : ty + actual_stamp_h, tx : tx + actual_stamp_w, 3
                    ],
                    stamp_crop[:, :, 3],
                )

            output.append(
                Sample(image=image_rgba, annotation=sample.annotation)
            )
        return output


class HandwritingTransform(BaseTransform):
    """Simulates handwritten notes or marks on the document.

    The algorithm uses Pillow to draw randomly selected text with a
    handwriting font at a random position.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["handwriting_transform", "HandwritingTransform"] = (
        "HandwritingTransform"
    )
    font_path: str
    texts: list[str] = Field(
        default_factory=lambda: ["Checked", "Approved", "Error", "Wait"]
    )
    probability: float = 0.3
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the handwriting transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            h, w = sample.image.shape[:2]
            img_pil = Image.fromarray(sample.image)
            draw = ImageDraw.Draw(img_pil)

            text = self.rng.choice(self.texts)
            font: ImageFont.FreeTypeFont | ImageFont.ImageFont
            try:
                font = ImageFont.truetype(
                    self.font_path, self.rng.randint(20, 40)
                )
            except OSError:
                font = ImageFont.load_default()

            tx = self.rng.randint(min(50, w), max(50, w - 200))
            ty = self.rng.randint(min(50, h), max(50, h - 100))
            color = (
                self.rng.randint(0, 50),
                self.rng.randint(0, 50),
                self.rng.randint(150, 255),
            )

            draw.text((tx, ty), text, font=font, fill=color)

            output.append(
                Sample(image=np.array(img_pil), annotation=sample.annotation)
            )
        return output


class CircleTransform(BaseTransform):
    """Draws a hand-drawn circle or ellipse around a text line.

    The algorithm selects a random text line and draws multiple overlapping,
    slightly jittered arcs in blue ink.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["circle_transform", "CircleTransform"] = "CircleTransform"
    probability: float = 0.4
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the circle transformation."""
        output = []
        rgba_channels = 4
        for sample in samples:
            if (
                self.rng.random() > self.probability
                or not sample.annotation.lines
            ):
                output.append(sample)
                continue

            img = sample.image.copy()
            line = self.rng.choice(sample.annotation.lines)
            rect = cv2.boundingRect(line.bbox.astype(np.int32))
            x, y, w, h = rect

            center = (x + w // 2, y + h // 2)
            axes = (int(w * 0.8), int(h * 1.2))
            angle = self.rng.randint(-10, 10)

            color_rgb = (
                self.rng.randint(0, 50),
                self.rng.randint(0, 100),
                self.rng.randint(180, 255),
            )
            color = (
                (*color_rgb, 200)
                if img.shape[2] == rgba_channels
                else color_rgb
            )

            for _ in range(2):
                jitter = self.rng.randint(-3, 3)
                cv2.ellipse(
                    img,
                    (center[0] + jitter, center[1] + jitter),
                    (axes[0] + jitter, axes[1] + jitter),
                    angle,
                    0,
                    360,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            output.append(Sample(image=img, annotation=sample.annotation))
        return output
