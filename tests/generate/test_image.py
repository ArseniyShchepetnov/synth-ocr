"""Tests for synthetic image generation."""

from unittest.mock import patch

from synthocr.generate.image import (
    FontConfig,
    SynthImageGenerator,
    font_configs_to_probability,
)


def test_document_generator_no_fonts() -> None:
    """Test document generator fallback to default font."""
    # Should fallback to default font
    gen = SynthImageGenerator(font_config=[])
    sample = gen.generate(["Hello World", "Test Line"])

    # DocumentGenerator returns RGBA (4 channels)
    assert sample.image.shape == (1024, 1024, 4)
    assert len(sample.annotation.lines) >= 2
    assert any(line.text == "Hello World" for line in sample.annotation.lines)


def test_font_configs_to_probability() -> None:
    """Test calculation of font probabilities from configurations."""
    font_configs = [
        FontConfig(path="font_en_1.ttf", priority=10, languages=["en"]),
        FontConfig(path="font_en_2.ttf", priority=30, languages=["en"]),
        FontConfig(path="font_ka.ttf", priority=10, languages=["ka"]),
        FontConfig(path="font_universal.ttf", priority=5, languages=[]),
    ]

    probs = font_configs_to_probability(font_configs)

    # English probabilities
    assert "en" in probs
    assert probs["en"]["font_en_1.ttf"] == 0.25
    assert probs["en"]["font_en_2.ttf"] == 0.75

    # Georgian probabilities
    assert "ka" in probs
    assert probs["ka"]["font_ka.ttf"] == 1.0

    # Universal probabilities
    assert "__universal__" in probs
    assert probs["__universal__"]["font_universal.ttf"] == 1.0


def test_select_font_logic() -> None:
    """Test _select_font respects language mappings and fallback."""
    font_configs = [
        FontConfig(path="/mock/font_en.ttf", priority=10, languages=["en"]),
        FontConfig(path="/mock/font_ka.ttf", priority=10, languages=["ka"]),
        FontConfig(path="/mock/font_univ.ttf", priority=10, languages=[]),
    ]

    gen = SynthImageGenerator(font_config=font_configs)

    # Mock Path.exists to return True for all mock fonts
    with patch("synthocr.generate.image.Path.exists", return_value=True):
        # 1. Test specific language
        font = gen._select_font("en")  # noqa: SLF001
        assert font == "/mock/font_en.ttf"

        font = gen._select_font("ka")  # noqa: SLF001
        assert font == "/mock/font_ka.ttf"

        # 2. Test fallback to universal for unknown language
        font = gen._select_font("fr")  # noqa: SLF001
        assert font == "/mock/font_univ.ttf"

        # 3. Test fallback to universal for None language
        font = gen._select_font(None)  # noqa: SLF001
        assert font == "/mock/font_univ.ttf"


def test_select_font_no_universal_fallback() -> None:
    """Test that _select_font returns None if no mapping exists."""
    font_configs = [
        FontConfig(path="/mock/font_en.ttf", priority=10, languages=["en"]),
    ]

    gen = SynthImageGenerator(font_config=font_configs)

    with patch("synthocr.generate.image.Path.exists", return_value=True):
        # Unknown language with no universal font should return None
        font = gen._select_font("ka")  # noqa: SLF001
        assert font is None
