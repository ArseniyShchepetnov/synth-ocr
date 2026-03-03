"""Tests for structured document generation."""

from synthocr.generate.structured_document import (
    Column,
    StructuredDocumentGenerator,
    StructuredRow,
)


def test_structured_document_generator_basic() -> None:
    """Test basic structured document generation."""
    gen = StructuredDocumentGenerator(image_size=(512, 512), padding=20)

    rows = [
        StructuredRow(
            columns=[
                Column(text="Name", weight=1.0),
                Column(text="Value", weight=1.0, align="right"),
            ],
            font_size=20,
        ),
        StructuredRow(
            columns=[
                Column(text="Total", weight=2.0, align="center"),
            ],
            font_size=24,
            is_bold=True,
        ),
    ]

    sample = gen.generate(rows)

    assert sample.image.shape == (512, 512, 4)
    # We have 3 columns total across 2 rows
    assert len(sample.annotation.lines) == 3

    texts = [line.text for line in sample.annotation.lines]
    assert "Name" in texts
    assert "Value" in texts
    assert "Total" in texts


def test_structured_document_generator_overflow() -> None:
    """Test that generator stops when content exceeds image height."""
    gen = StructuredDocumentGenerator(image_size=(100, 100), padding=10)

    # Add many rows that will definitely overflow 100px
    rows = [
        StructuredRow(columns=[Column(text=f"Row {i}")], font_size=30)
        for i in range(10)
    ]

    sample = gen.generate(rows)

    # Should have fewer than 10 rows due to overflow
    assert len(sample.annotation.lines) < 10
    assert len(sample.annotation.lines) > 0
