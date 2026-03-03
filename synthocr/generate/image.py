"""Synthetic image generation logic for OCR datasets."""

import random
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from synthocr.generate.transforms import Annotation, Sample, TextLine


class FontConfig(BaseModel):
    """Configuration for a single font."""

    path: str
    priority: float = 1.0
    languages: list[str] = Field(default_factory=list)


def font_configs_to_probability(
    font_configs: list[FontConfig],
) -> dict[str, dict[str, float]]:
    """Get fonts probabilities for each languages from configuration."""
    lang_to_fonts: dict[str, list[FontConfig]] = {}
    universal_fonts: list[FontConfig] = []

    for config in font_configs:
        if not config.languages:
            universal_fonts.append(config)
        for lang in config.languages:
            if lang not in lang_to_fonts:
                lang_to_fonts[lang] = []
            lang_to_fonts[lang].append(config)

    result: dict[str, dict[str, float]] = {}

    # Helper to calculate probabilities for a list of configs
    def calc_probs(configs: list[FontConfig]) -> dict[str, float]:
        total_priority = sum(c.priority for c in configs)
        if total_priority > 0:
            return {c.path: c.priority / total_priority for c in configs}
        return {c.path: 1.0 / len(configs) for c in configs}

    # Pre-calculate for each language found in config
    for lang, configs in lang_to_fonts.items():
        result[lang] = calc_probs(configs)

    # Add a special key for universal fonts
    if universal_fonts:
        result["__universal__"] = calc_probs(universal_fonts)

    return result


class SynthImageGenerator(BaseModel):
    """Generates synthetic document images by scattering text blocks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    font_config: list[FontConfig] = Field(default_factory=list)
    image_size: tuple[int, int] = (1024, 1024)
    min_font_size: int = 12
    max_font_size: int = 32
    min_blocks: int = 4
    max_blocks: int = 10
    margin: int = 10
    min_text_length: int = 5

    _font_probabilities: dict[str, dict[str, float]] = PrivateAttr(
        default_factory=dict
    )

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Initialize font probabilities."""
        self._font_probabilities = font_configs_to_probability(
            self.font_config
        )

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

        # 1. Try to use pre-calculated probabilities for the
        # specific language
        if language and language in self._font_probabilities:
            lang_fonts = self._font_probabilities[language]
            paths = list(lang_fonts.keys())
            probs = list(lang_fonts.values())

            # Filter by existence at runtime
            valid_indices = [
                i for i, p in enumerate(paths) if Path(p).exists()
            ]
            if valid_indices:
                current_paths = [paths[i] for i in valid_indices]
                current_probs = [probs[i] for i in valid_indices]
                return random.choices(  # noqa: S311
                    current_paths, weights=current_probs, k=1
                )[0]

        # 2. Fallback to universal fonts (no languages specified in config)
        if "__universal__" in self._font_probabilities:
            univ_fonts = self._font_probabilities["__universal__"]
            paths = list(univ_fonts.keys())
            probs = list(univ_fonts.values())

            valid_indices = [
                i for i, p in enumerate(paths) if Path(p).exists()
            ]
            if valid_indices:
                current_paths = [paths[i] for i in valid_indices]
                current_probs = [probs[i] for i in valid_indices]
                return random.choices(  # noqa: S311
                    current_paths, weights=current_probs, k=1
                )[0]

        # 3. If still nothing found, return None to trigger
        # default font fallback
        return None

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
