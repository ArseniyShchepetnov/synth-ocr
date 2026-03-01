"""Synthetic image generation logic for OCR datasets."""

import random
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from synthocr.generate.transforms import Annotation, Sample, TextLine


class SynthImageGenerator(BaseModel):
    """Generates synthetic document images by scattering text blocks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    font_config: list[dict] = Field(default_factory=list)
    image_size: tuple[int, int] = (1024, 1024)
    min_font_size: int = 12
    max_font_size: int = 32
    min_blocks: int = 4
    max_blocks: int = 10
    margin: int = 10
    min_text_length: int = 5

    def _get_random_color(
        self, *, bright: bool = True
    ) -> tuple[int, int, int]:
        """Get a random RGB color."""
        if bright:
            return (
                random.randint(200, 255),  # noqa: S311
                random.randint(200, 255),  # noqa: S311
                random.randint(200, 255),  # noqa: S311
            )
        return (
            random.randint(0, 80),  # noqa: S311
            random.randint(0, 80),  # noqa: S311
            random.randint(0, 80),  # noqa: S311
        )

    def _select_font(self, language: str | None) -> str | None:
        """Select a font path based on language and priority."""
        if not self.font_config:
            return None

        # Filter fonts by language
        available_fonts = []
        if language:
            available_fonts = [
                f
                for f in self.font_config
                if language in f.get("languages", [])
                and Path(f["path"]).exists()
            ]

        # Fallback to any available font if language specific not found
        if not available_fonts:
            available_fonts = [
                f for f in self.font_config if Path(f["path"]).exists()
            ]

        if not available_fonts:
            return None

        paths = [f["path"] for f in available_fonts]
        priorities = [f.get("priority", 1) for f in available_fonts]

        return random.choices(paths, weights=priorities, k=1)[0]  # noqa: S311

    def _load_font(
        self, path: str | None, size: int, language: str | None
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load a font or fallback to default."""
        if path:
            try:
                font = ImageFont.truetype(path, size)
                logger.debug(f"Using font '{path}' for language '{language}'")
            except OSError as e:
                logger.warning(f"Failed to load font {path}: {e}")
            else:
                return font

        logger.debug(f"No font found for language '{language}', using default")
        return ImageFont.load_default()

    def _check_overlap(
        self,
        new_rect: tuple[float, float, float, float],
        occupied: list[tuple[float, float, float, float]],
    ) -> bool:
        """Check if a new rectangle overlaps with any occupied ones."""
        nx1, ny1, nx2, ny2 = new_rect
        for ox1, oy1, ox2, oy2 in occupied:
            if not (nx2 < ox1 or nx1 > ox2 or ny2 < oy1 or ny1 > oy2):
                return True
        return False

    def _render_text_block(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        position: tuple[int, int, int],
    ) -> tuple[list[TextLine], tuple[float, float, float, float]] | None:
        """Render a text block onto the image."""
        rx, ry, region_width = position

        # Text wrapping
        words = text.split()
        wrapped_lines = []
        current_line: list[str] = []
        for word in words:
            test_line = " ".join([*current_line, word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] < region_width:
                current_line.append(word)
            elif current_line:
                wrapped_lines.append(" ".join(current_line))
                current_line = [word]
            else:
                wrapped_lines.append(word)
                current_line = []
        if current_line:
            wrapped_lines.append(" ".join(current_line))

        text_color = self._get_random_color(bright=False)
        current_y = float(ry)

        block_min_x, block_min_y = float(rx), float(ry)
        block_max_x, block_max_y = float(rx), float(ry)

        block_lines = []
        for line_text in wrapped_lines:
            bbox = draw.textbbox((rx, current_y), line_text, font=font)

            if bbox[3] > self.image_size[1] - 20:
                break

            draw.text((rx, current_y), line_text, font=font, fill=text_color)

            points = np.array(
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                ],
                dtype=np.float32,
            )

            block_lines.append(TextLine(text=line_text, bbox=points))

            block_min_x = min(block_min_x, bbox[0])
            block_min_y = min(block_min_y, bbox[1])
            block_max_x = max(block_max_x, bbox[2])
            block_max_y = max(block_max_y, bbox[3])

            current_y = bbox[3] + 2

        if not block_lines:
            return None

        occupied_bbox = (
            block_min_x - self.margin,
            block_min_y - self.margin,
            block_max_x + self.margin,
            block_max_y + self.margin,
        )
        return block_lines, occupied_bbox

    def generate(
        self, text_samples: list[str], language: str | None = None
    ) -> Sample:
        """Generate a single sample with text blocks."""
        # Create a transparent background
        image = Image.new("RGBA", self.image_size, color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        lines: list[TextLine] = []
        occupied_bboxes: list[tuple[float, float, float, float]] = []

        # Select a subset of samples to place as scattered blocks
        num_blocks = min(
            len(text_samples),
            random.randint(self.min_blocks, self.max_blocks),  # noqa: S311
        )
        selected_samples = random.sample(text_samples, num_blocks)

        for text in selected_samples:
            if not text.strip() or len(text) < self.min_text_length:
                continue

            font_size = random.randint(  # noqa: S311
                self.min_font_size, self.max_font_size
            )

            font_path = self._select_font(language)
            font = self._load_font(font_path, font_size, language)

            # Try several times to find a non-overlapping position
            best_pos = None
            # We need to estimate height. Wrapped lines haven't been calculated
            # yet, but we can guess based on text length and font size.
            est_height = (len(text) / 20) * font_size * 1.5 + 40

            for _ in range(20):
                # Random width between 150 and 500
                region_width = random.randint(150, 500)  # noqa: S311
                # Random start position
                rx = random.randint(  # noqa: S311
                    20, max(21, self.image_size[0] - region_width - 20)
                )
                ry = random.randint(  # noqa: S311
                    20, max(21, self.image_size[1] - int(est_height) - 20)
                )

                # Proper rectangle intersection check
                new_rect = (rx, ry, rx + region_width, ry + est_height)
                if not self._check_overlap(new_rect, occupied_bboxes):
                    best_pos = (rx, ry, region_width)
                    break

            if not best_pos:
                continue

            result = self._render_text_block(draw, text, font, best_pos)
            if result:
                block_lines, occ_bbox = result
                lines.extend(block_lines)
                occupied_bboxes.append(occ_bbox)

        return Sample(
            image=np.array(image), annotation=Annotation(lines=lines)
        )
