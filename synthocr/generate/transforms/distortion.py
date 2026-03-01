"""Distortion transformations for OCR dataset generation."""

import random
from functools import cached_property
from typing import Literal

import cv2
import numpy as np
from pydantic import ConfigDict

from synthocr.generate.transforms.base import (
    Annotation,
    BaseTransform,
    Sample,
    TextLine,
    sample_int_param,
    sample_param,
)


class RotationTransform(BaseTransform):
    """Applies random rotation to samples.

    The algorithm computes a rotation matrix based on a sampled angle and
    applies it to both the image (using affine warping with transparent
    borders) and the bounding box coordinates.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["rotation_transform", "RotationTransform"] = (
        "RotationTransform"
    )
    max_angle: float | list[float]
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the rotation transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            max_angle = sample_param(self.max_angle, self.rng)
            angle = self.rng.uniform(-max_angle, max_angle)
            h, w = sample.image.shape[:2]
            center = (w / 2, h / 2)

            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(
                sample.image, matrix, (w, h), borderValue=(0, 0, 0, 0)
            )

            new_lines = []
            for line in sample.annotation.lines:
                ones = np.ones(shape=(len(line.bbox), 1))
                points_ones = np.hstack([line.bbox, ones])
                new_bbox = matrix.dot(points_ones.T).T
                new_lines.append(TextLine(text=line.text, bbox=new_bbox))
            output.append(
                Sample(
                    image=rotated_image, annotation=Annotation(lines=new_lines)
                )
            )
        return output


class PerspectiveTransform(BaseTransform):
    """Applies random perspective distortion.

    The algorithm maps the corners of the image to new random positions
    within a defined distortion scale and applies the resulting perspective
    matrix to both the image and the text line bounding boxes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["perspective_transform", "PerspectiveTransform"] = (
        "PerspectiveTransform"
    )
    distortion_scale: float | list[float] = 0.2
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the perspective transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            dist_scale = sample_param(self.distortion_scale, self.rng)
            h, w = sample.image.shape[:2]
            dist = dist_scale * min(h, w)
            pts1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            pts2 = np.array(
                [
                    [self.rng.uniform(0, dist), self.rng.uniform(0, dist)],
                    [w - self.rng.uniform(0, dist), self.rng.uniform(0, dist)],
                    [
                        w - self.rng.uniform(0, dist),
                        h - self.rng.uniform(0, dist),
                    ],
                    [self.rng.uniform(0, dist), h - self.rng.uniform(0, dist)],
                ],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped_image = cv2.warpPerspective(
                sample.image, matrix, (w, h), borderValue=(0, 0, 0, 0)
            )
            new_lines = []
            for line in sample.annotation.lines:
                points = line.bbox.reshape(-1, 1, 2)
                new_bbox = cv2.perspectiveTransform(points, matrix).reshape(
                    -1, 2
                )
                new_lines.append(TextLine(text=line.text, bbox=new_bbox))
            output.append(
                Sample(
                    image=warped_image, annotation=Annotation(lines=new_lines)
                )
            )
        return output


class ShearTransform(BaseTransform):
    """Applies random shear distortion to the image.

    The algorithm generates an affine transformation matrix representing
    horizontal and vertical shearing and applies it to the image and its
    annotations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["shear_transform", "ShearTransform"] = "ShearTransform"
    max_shear: float | list[float] = 0.1
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the shear transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            max_s = sample_param(self.max_shear, self.rng)
            h, w = sample.image.shape[:2]
            shear_x = self.rng.uniform(-max_s, max_s)
            shear_y = self.rng.uniform(-max_s, max_s)
            matrix = np.array(
                [[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32
            )
            warped_image = cv2.warpAffine(
                sample.image, matrix, (w, h), borderValue=(0, 0, 0, 0)
            )
            new_lines = []
            for line in sample.annotation.lines:
                ones = np.ones(shape=(len(line.bbox), 1))
                points_ones = np.hstack([line.bbox, ones])
                new_bbox = matrix.dot(points_ones.T).T
                new_lines.append(TextLine(text=line.text, bbox=new_bbox))
            output.append(
                Sample(
                    image=warped_image, annotation=Annotation(lines=new_lines)
                )
            )
        return output


class CrushedDocumentTransform(BaseTransform):
    """Simulates a crushed or folded document.

    The algorithm applies a coarse elastic deformation using Cubic
    interpolation remapping and adds random linear fold lines to the RGB
    channels by adjusting pixel intensity along sampled vectors.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["crushed_document_transform", "CrushedDocumentTransform"] = (
        "CrushedDocumentTransform"
    )
    distortion_scale: float | list[float] = 0.03
    num_folds: int | list[int] = 8
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the crushed document transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            dist_scale = sample_param(self.distortion_scale, self.rng)
            folds = sample_int_param(self.num_folds, self.rng)

            image = sample.image.copy()
            h, w = image.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            cw, ch = 8, 8
            dx_coarse = np.array(
                [
                    [
                        self.rng.uniform(-dist_scale * w, dist_scale * w)
                        for _ in range(cw)
                    ]
                    for _ in range(ch)
                ]
            )
            dy_coarse = np.array(
                [
                    [
                        self.rng.uniform(-dist_scale * h, dist_scale * h)
                        for _ in range(cw)
                    ]
                    for _ in range(ch)
                ]
            )
            dx = cv2.resize(dx_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
            dy = cv2.resize(dy_coarse, (w, h), interpolation=cv2.INTER_CUBIC)
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            warped_image = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for _ in range(folds):
                p1 = (self.rng.randint(0, w - 1), self.rng.randint(0, h - 1))
                p2 = (self.rng.randint(0, w - 1), self.rng.randint(0, h - 1))
                shift = self.rng.randint(-40, 40)
                overlay = warped_image[:, :, :3].astype(np.int16)
                cv2.line(
                    overlay,
                    p1,
                    p2,
                    (shift, shift, shift),
                    thickness=self.rng.randint(1, 3),
                )
                warped_image[:, :, :3] = np.clip(overlay, 0, 255).astype(
                    np.uint8
                )

            new_lines = []
            for line in sample.annotation.lines:
                new_bbox = []
                for pt in line.bbox:
                    px, py = (
                        int(np.clip(pt[0], 0, w - 1)),
                        int(np.clip(pt[1], 0, h - 1)),
                    )
                    new_px = pt[0] - dx[py, px]
                    new_py = pt[1] - dy[py, px]
                    new_bbox.append([new_px, new_py])
                new_lines.append(
                    TextLine(text=line.text, bbox=np.array(new_bbox))
                )
            output.append(
                Sample(
                    image=warped_image, annotation=Annotation(lines=new_lines)
                )
            )
        return output


class WaveTransform(BaseTransform):
    """Applies a sine wave distortion to the image.

    The algorithm computes a pixel mapping based on a sine function along the
    specified direction (vertical or horizontal) and uses remapping for the
    image. Bounding boxes are updated by sampling points along their edges
    and fitting a minimum area rectangle to the transformed shape.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["wave_transform", "WaveTransform"] = "WaveTransform"
    amplitude: float | list[float] = 10.0
    period: float | list[float] = 200.0
    direction: str = "vertical"
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the wave transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            amp = sample_param(self.amplitude, self.rng)
            per = sample_param(self.period, self.rng)

            h, w = sample.image.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            if self.direction == "vertical":
                dy = amp * np.sin(2 * np.pi * x / per)
                dx = np.zeros_like(dy)
            else:
                dx = amp * np.sin(2 * np.pi * y / per)
                dy = np.zeros_like(dx)

            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)

            warped_image = cv2.remap(
                sample.image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            new_lines = []
            for line in sample.annotation.lines:
                pts = line.bbox
                sampled_points = []
                for i in range(len(pts)):
                    p1 = pts[i]
                    p2 = pts[(i + 1) % len(pts)]
                    num_steps = 20
                    for t in np.linspace(0, 1, num_steps, endpoint=False):
                        sampled_points.append(p1 * (1 - t) + p2 * t)

                transformed_pts = []
                for pt in sampled_points:
                    px, py = pt[0], pt[1]
                    if self.direction == "vertical":
                        new_px = px
                        new_py = py - amp * np.sin(2 * np.pi * px / per)
                    else:
                        new_px = px - amp * np.sin(2 * np.pi * py / per)
                        new_py = py
                    transformed_pts.append([new_px, new_py])

                pts_array = np.array(transformed_pts).astype(np.float32)
                rect = cv2.minAreaRect(pts_array)
                box = cv2.boxPoints(rect)
                new_rect = np.array(box, dtype=np.float32)

                new_lines.append(TextLine(text=line.text, bbox=new_rect))

            output.append(
                Sample(
                    image=warped_image, annotation=Annotation(lines=new_lines)
                )
            )
        return output


class CylindricalTransform(BaseTransform):
    """Simulates paper curvature or cylindrical wrapping.

    The algorithm applies a parabolic vertical mapping to simulate the
    warping of paper when held or placed on a curved surface. Bounding
    boxes are adjusted and fitted with minimum area rectangles to maintain
    structural integrity.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["cylindrical_transform", "CylindricalTransform"] = (
        "CylindricalTransform"
    )
    curvature: float = 0.0001
    probability: float = 0.5
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the cylindrical transformation."""
        output = []
        for sample in samples:
            if self.rng.random() > self.probability:
                output.append(sample)
                continue

            h, w = sample.image.shape[:2]
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            k = self.rng.uniform(0, self.curvature)
            dy = k * (x - w / 2) ** 2

            map_x = x.astype(np.float32)
            map_y = (y + dy).astype(np.float32)

            warped_image = cv2.remap(
                sample.image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            new_lines = []
            for line in sample.annotation.lines:
                new_poly = []
                for pt in line.bbox:
                    px, py = pt[0], pt[1]
                    new_py = py - k * (px - w / 2) ** 2
                    new_poly.append([px, new_py])

                poly_array = np.array(new_poly).astype(np.float32)
                rect = cv2.minAreaRect(poly_array)
                box = cv2.boxPoints(rect)
                new_lines.append(
                    TextLine(
                        text=line.text, bbox=np.array(box, dtype=np.float32)
                    )
                )

            output.append(
                Sample(
                    image=warped_image, annotation=Annotation(lines=new_lines)
                )
            )
        return output
