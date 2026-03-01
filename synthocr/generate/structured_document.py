"""Structured document generation logic for OCR datasets."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synthocr.generate.transforms import Annotation, Sample, TextLine


@dataclass
class Column:
    """Represents a single column in a structured row."""

    text: str
    weight: float = 1.0
    align: str = "left"  # "left", "center", "right"


@dataclass
class StructuredRow:
    """Represents a row containing one or more columns."""

    columns: list[Column]
    font_size: int = 24
    is_bold: bool = False
    margin_top: int = 5
    margin_bottom: int = 5


class StructuredDocumentGenerator:
    """Generates documents with structured rows and columns."""

    def __init__(
        self,
        font_config: list[dict] | None = None,
        image_size: tuple[int, int] = (1024, 1024),
        padding: int = 40,
    ):
        """Initialize the structured document generator."""
        self.font_config = font_config or []
        self.image_size = image_size
        self.padding = padding

    def _select_font(
        self,
        _language: str | None = None,
        *,
        _is_bold: bool = False,
    ) -> str | None:
        """Select a font path based on language and priority."""
        if not self.font_config:
            return None

        # Filter by language and weight if possible
        # For now, we assume languages=["en", "ka", "ru"] for universal fonts
        available = [f for f in self.font_config if Path(f["path"]).exists()]

        if not available:
            return None

        paths = [f["path"] for f in available]
        priorities = [f.get("priority", 1) for f in available]

        return random.choices(paths, weights=priorities, k=1)[0]  # noqa: S311

    def _get_random_color(
        self, *, bright: bool = False
    ) -> tuple[int, int, int]:
        """Get a random RGB color."""
        if bright:
            return (
                random.randint(200, 255),  # noqa: S311
                random.randint(200, 255),  # noqa: S311
                random.randint(200, 255),  # noqa: S311
            )
        return (
            random.randint(0, 60),  # noqa: S311
            random.randint(0, 60),  # noqa: S311
            random.randint(0, 60),  # noqa: S311
        )

    def generate(self, rows: list[StructuredRow]) -> Sample:
        """Generate a structured document from a list of rows."""
        # Create transparent background
        image = Image.new("RGBA", self.image_size, color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        lines: list[TextLine] = []
        current_y = float(self.padding)
        content_width = self.image_size[0] - 2 * self.padding

        text_color = self._get_random_color(bright=False)

        for row in rows:
            # Select font for the row
            font_path = self._select_font()
            font: ImageFont.FreeTypeFont | ImageFont.ImageFont
            if not font_path:
                font = ImageFont.load_default()
            else:
                try:
                    font = ImageFont.truetype(font_path, row.font_size)
                except OSError:
                    font = ImageFont.load_default()

            current_y += row.margin_top

            # Calculate column widths based on weights
            total_weight = sum(c.weight for c in row.columns)
            current_x = float(self.padding)

            for col in row.columns:
                col_width = (col.weight / total_weight) * content_width

                # Render text with alignment
                bbox = draw.textbbox((0, 0), col.text, font=font)
                text_w = bbox[2] - bbox[0]

                if col.align == "center":
                    tx = current_x + (col_width - text_w) / 2
                elif col.align == "right":
                    tx = current_x + (col_width - text_w)
                else:
                    tx = current_x

                # Draw text
                draw.text(
                    (tx, current_y), col.text, font=font, fill=text_color
                )

                # Save annotation
                # Use actual rendered bbox for precision
                final_bbox = draw.textbbox(
                    (tx, current_y), col.text, font=font
                )
                points = np.array(
                    [
                        [final_bbox[0], final_bbox[1]],
                        [final_bbox[2], final_bbox[1]],
                        [final_bbox[2], final_bbox[3]],
                        [final_bbox[0], final_bbox[3]],
                    ],
                    dtype=np.float32,
                )

                lines.append(TextLine(text=col.text, bbox=points))

                current_x += col_width

            # Move to next line
            current_y += row.font_size + row.margin_bottom

            if current_y > self.image_size[1] - self.padding:
                break

        return Sample(
            image=np.array(image), annotation=Annotation(lines=lines)
        )

    def draw_separator(
        self, image: Image.Image, y: int, pattern: str = "dotted"
    ) -> int:
        """Draw a separator line on the document."""
        draw = ImageDraw.Draw(image)
        x1, x2 = self.padding, self.image_size[0] - self.padding
        color = self._get_random_color(bright=False)

        if pattern == "dotted":
            for x in range(x1, x2, 10):
                draw.line([(x, y), (x + 4, y)], fill=color, width=2)
        else:
            draw.line([(x1, y), (x2, y)], fill=color, width=2)

        return y + 10
