"""Batch mixing transformations for OCR dataset generation."""

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
)


class MixBatchTransform(BaseTransform):
    """Generates synthetic samples by mixing text lines from multiple sources.

    The algorithm selects a subset of samples from the batch, extracts their
    text lines using polygon masks, and places them onto a new transparent
    canvas. It ensures no overlaps between placed elements and uses alpha
    compositing for realistic blending.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["mix_batch_transform", "MixBatchTransform"] = (
        "MixBatchTransform"
    )
    m: int
    k: int
    min_lines: int = 10
    max_lines: int = 20
    margin: int = 5
    probability: float = 1.0
    seed: int | None = None

    @cached_property
    def rng(self) -> random.Random:
        """Random number generator initialized once."""
        return random.Random(self.seed)  # noqa: S311

    def _check_overlap(
        self,
        new_rect: tuple[int, int, int, int],
        occupied: list[tuple[int, int, int, int]],
    ) -> bool:
        """Check if a new rectangle overlaps with any occupied ones."""
        nx1, ny1, nx2, ny2 = new_rect
        for ox, oy, ocw, och in occupied:
            if not (nx2 < ox or nx1 > ox + ocw or ny2 < oy or ny1 > oy + och):
                return True
        return False

    def _prepare_crop(
        self, src_img: np.ndarray, src_line: TextLine
    ) -> tuple[np.ndarray, int, int] | None:
        """Extract and mask the text line from the source image."""
        bbox_int = src_line.bbox.astype(np.int32)
        rx, ry, rw, rh = cv2.boundingRect(bbox_int)

        if rw <= 0 or rh <= 0:
            return None

        src_h, src_w = src_img.shape[:2]
        ry_start, ry_end = max(0, ry), min(ry + rh, src_h)
        rx_start, rx_end = max(0, rx), min(rx + rw, src_w)

        if ry_end <= ry_start or rx_end <= rx_start:
            return None

        crop = src_img[ry_start:ry_end, rx_start:rx_end].copy()
        crh, crw = crop.shape[:2]

        mask = np.zeros((crh, crw), dtype=np.uint8)
        poly_relative = (
            src_line.bbox - np.array([rx_start, ry_start])
        ).astype(np.int32)
        cv2.fillPoly(mask, [poly_relative], 255)

        crop[:, :, 3] = (
            crop[:, :, 3].astype(np.float32) * (mask / 255.0)
        ).astype(np.uint8)

        return crop, rx_start, ry_start

    def __call__(self, samples: list[Sample]) -> list[Sample]:
        """Apply the batch mixing transformation."""
        if not samples or self.rng.random() > self.probability:
            return samples

        n = len(samples)
        h, w = samples[0].image.shape[:2]
        output = []
        rgba_channels = 4
        rgb_channels = 3

        for _ in range(self.k):
            m_val = min(self.m, n)
            selected_sources = self.rng.sample(samples, m_val)

            all_source_lines = []
            for s in selected_sources:
                for line in s.annotation.lines:
                    all_source_lines.append((s.image, line))

            if not all_source_lines:
                output.append(self.rng.choice(samples))
                continue

            new_image = np.zeros((h, w, rgba_channels), dtype=np.uint8)
            new_annotation_lines: list[TextLine] = []
            occupied_bboxes: list[tuple[int, int, int, int]] = []

            num_lines_to_place = min(
                len(all_source_lines),
                self.rng.randint(self.min_lines, self.max_lines),
            )
            lines_to_place = self.rng.sample(
                all_source_lines, num_lines_to_place
            )

            for src_img, src_line in lines_to_place:
                result = self._prepare_crop(src_img, src_line)
                if result is None:
                    continue
                crop, rx_start, ry_start = result
                crh, crw = crop.shape[:2]

                best_pos = None
                for _ in range(30):
                    tx = self.rng.randint(
                        self.margin,
                        max(self.margin + 1, w - crw - self.margin),
                    )
                    ty = self.rng.randint(
                        self.margin,
                        max(self.margin + 1, h - crh - self.margin),
                    )

                    new_rect = (
                        tx - self.margin,
                        ty - self.margin,
                        tx + crw + self.margin,
                        ty + crh + self.margin,
                    )
                    if not self._check_overlap(new_rect, occupied_bboxes):
                        best_pos = (tx, ty)
                        break

                if best_pos:
                    tx, ty = best_pos
                    alpha_src = crop[:, :, 3] / 255.0
                    target_region = new_image[ty : ty + crh, tx : tx + crw]
                    for c in range(rgb_channels):
                        target_region[:, :, c] = (
                            alpha_src * crop[:, :, c]
                            + (1.0 - alpha_src) * target_region[:, :, c]
                        ).astype(np.uint8)

                    alpha_dst = target_region[:, :, 3] / 255.0
                    target_region[:, :, 3] = (
                        (alpha_src + alpha_dst * (1.0 - alpha_src)) * 255
                    ).astype(np.uint8)

                    shift = np.array([tx - rx_start, ty - ry_start])
                    new_bbox = src_line.bbox + shift

                    new_annotation_lines.append(
                        TextLine(text=src_line.text, bbox=new_bbox)
                    )
                    occupied_bboxes.append((tx, ty, crw, crh))

            output.append(
                Sample(
                    image=new_image,
                    annotation=Annotation(lines=new_annotation_lines),
                )
            )

        return output
