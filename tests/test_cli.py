"""Tests for CLI configuration loading."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from click.testing import CliRunner

from synthocr.cli import generate


def test_generate_loads_config_correctly(tmp_path: Path) -> None:
    """Test generate command loads parameters from config file."""
    config_data: dict[str, Any] = {
        "source": {
            "type": "wikipedia",
            "languages": [{"code": "en", "priority": 1.0}],
        },
        "document_generator": {
            "min_blocks": 5,
            "max_blocks": 15,
            "margin": 20,
            "font_config": [
                {
                    "path": "/mock/font_0.ttf",
                    "priority": 10,
                    "languages": ["en"],
                },
                {
                    "path": "/mock/font_1.ttf",
                    "priority": 10,
                    "languages": ["ka"],
                },
            ],
        },
        "min_text_samples": 5,
        "max_text_samples": 10,
        "transforms": [],
    }

    config_file = tmp_path / "config.json"
    with config_file.open("w") as f:
        json.dump(config_data, f)

    runner = CliRunner()

    # We mock WikipediaSource and GenerationPipeline.run to avoid network
    with (
        patch("synthocr.cli.WikipediaSource") as mock_source_cls,
        patch("synthocr.cli.SynthImageGenerator") as mock_gen_cls,
        patch("synthocr.cli.GenerationPipeline") as mock_pipeline_cls,
        patch("os._exit"),
    ):
        result = runner.invoke(
            generate,
            [
                "--config",
                str(config_file),
                "--num-samples",
                "1",
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code == 0

        # Check if SynthImageGenerator was called with values from config
        mock_gen_cls.assert_called_once()
        _, kwargs = mock_gen_cls.call_args
        gen_kwargs: dict[str, Any] = kwargs

        # Assertions for generator
        doc_gen_config: dict[str, Any] = cast(
            "dict[str, Any]", config_data["document_generator"]
        )
        assert gen_kwargs["font_config"] == doc_gen_config["font_config"]
        assert gen_kwargs["min_blocks"] == doc_gen_config["min_blocks"]
        assert gen_kwargs["max_blocks"] == doc_gen_config["max_blocks"]
        assert gen_kwargs["margin"] == doc_gen_config["margin"]

        # Check source initialization
        mock_source_cls.assert_called_once()
        _, source_kwargs_raw = mock_source_cls.call_args
        source_kwargs: dict[str, Any] = source_kwargs_raw
        source_config: dict[str, Any] = cast(
            "dict[str, Any]", config_data["source"]
        )
        assert source_kwargs["languages"] == source_config["languages"]

        # Check pipeline initialization
        mock_pipeline_cls.assert_called_once()
        _, pipeline_kwargs_raw = mock_pipeline_cls.call_args
        pipeline_kwargs: dict[str, Any] = pipeline_kwargs_raw
        assert (
            pipeline_kwargs["min_text_samples"]
            == config_data["min_text_samples"]
        )
        assert (
            pipeline_kwargs["max_text_samples"]
            == config_data["max_text_samples"]
        )


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
