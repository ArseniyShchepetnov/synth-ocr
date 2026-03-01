"""Tests for document generation."""

from synthocr.generate.image import SynthImageGenerator


def test_document_generator_no_fonts() -> None:
    """Test document generator fallback to default font."""
    # Should fallback to default font
    gen = SynthImageGenerator(font_config=[])
    sample = gen.generate(["Hello World", "Test Line"])

    # DocumentGenerator returns RGBA (4 channels)
    assert sample.image.shape == (1024, 1024, 4)
    assert len(sample.annotation.lines) >= 2
    assert any(line.text == "Hello World" for line in sample.annotation.lines)
