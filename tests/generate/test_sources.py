"""Tests for text sources."""

from unittest.mock import MagicMock, patch

from synthocr.generate.sources import SourceLanguageConfig, WikipediaSource


def test_wikipedia_source_priorities() -> None:
    """Test WikipediaSource respects language priorities during sampling."""
    languages = [
        SourceLanguageConfig(code="en", priority=10.0),
        SourceLanguageConfig(code="ka", priority=1.0),
    ]

    # Mock datasets to avoid network calls and provide controlled data
    with patch("synthocr.generate.sources.load_dataset") as mock_load:
        # Create mock iterators that return different strings for each language
        mock_ds_en = MagicMock()
        mock_ds_en.__iter__.return_value = iter(
            [{"text": f"en_{i}"} for i in range(1100)]
        )

        mock_ds_ka = MagicMock()
        mock_ds_ka.__iter__.return_value = iter(
            [{"text": f"ka_{i}"} for i in range(1100)]
        )

        mock_load.side_effect = lambda _p, name, _s, _st: {
            "20231101.en": mock_ds_en,
            "20231101.ka": mock_ds_ka,
        }[name]

        source = WikipediaSource(languages=languages)

        # Sample many times and check distribution
        samples = [next(source.iter_samples()) for _ in range(1000)]

        en_count = sum(1 for s in samples if s.startswith("en_"))
        ka_count = sum(1 for s in samples if s.startswith("ka_"))

        # With 10:1 priority, we expect roughly 909 en and 91 ka.
        # Let's check it's at least significantly more English.
        assert en_count > ka_count * 5
        assert ka_count > 0


def test_wikipedia_source_get_text() -> None:
    """Test getting text for a specific language."""
    languages = [SourceLanguageConfig(code="en", priority=1.0)]

    with patch("synthocr.generate.sources.load_dataset") as mock_load:
        mock_ds = MagicMock()
        mock_ds.__iter__.return_value = iter([{"text": "hello"}])
        mock_load.return_value = mock_ds

        source = WikipediaSource(languages=languages)
        assert source.get_text("en") == "hello"
