"""Command line interface for the OCR dataset generator."""

import json
import os
import shutil
from pathlib import Path

import click
import decouple

from synthocr.generate.image import SynthImageGenerator
from synthocr.generate.pipeline import GenerationPipeline
from synthocr.generate.sources import WikipediaSource
from synthocr.generate.transforms import (
    SampleTransformPipeline,
    get_pipeline_schema,
)


@click.group()
@click.option(
    "--extra-transforms", multiple=True, help="Extra transform modules to load"
)
def cli(extra_transforms: list[str]):
    """OCR Dataset Generator CLI."""
    if extra_transforms:
        os.environ["OCR_EXTRA_TRANSFORMS"] = ",".join(extra_transforms)


@cli.command()
@click.option(
    "--output", type=click.Path(), default="output", help="Output directory"
)
@click.option(
    "--num-samples", type=int, default=10, help="Number of samples to generate"
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Path to transform pipeline config JSON",
)
def generate(
    output: str,
    num_samples: int,
    config_file: str | None,
):
    """Run the OCR dataset generation pipeline."""
    hf_token = decouple.config("HF_TOKEN", default=None)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    click.echo(f"Starting generation of {num_samples} samples...")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    config = {}
    if config_file:
        click.echo(f"Loading config from {config_file}")
        config_path = Path(config_file)
        with config_path.open() as f:
            config = json.load(f)

        shutil.copy2(config_file, output_path / "config.json")

    if isinstance(config, list):
        transform_pipeline = SampleTransformPipeline.from_config(config)
        source_langs = [{"code": "en", "priority": 1.0}]
    else:
        # Check for nested document_generator config
        doc_gen_config = config.get("document_generator", {})

        transforms_config = config.get("transforms", [])
        transform_pipeline = SampleTransformPipeline.from_config(
            transforms_config
        )

        # Source config handles languages
        source_config = config.get("source", {})
        source_langs = source_config.get(
            "languages",
            config.get("languages", [{"code": "en", "priority": 1.0}]),
        )

        # Prefer nested font_config if it exists, otherwise
        # fall back to top-level "fonts"
        font_config = doc_gen_config.get(
            "font_config", config.get("fonts", [])
        )

        langs_codes = [lang["code"] for lang in source_langs]
        click.echo(f"Loading Wikipedia source for languages: {langs_codes}")
        source = WikipediaSource(languages=source_langs)

    # Load generator parameters from nested doc_gen_config if available
    min_blocks = (
        doc_gen_config.get("min_blocks", config.get("min_blocks", 4))
        if not isinstance(config, list)
        else 4
    )
    max_blocks = (
        doc_gen_config.get("max_blocks", config.get("max_blocks", 10))
        if not isinstance(config, list)
        else 10
    )
    margin = (
        doc_gen_config.get("margin", config.get("margin", 10))
        if not isinstance(config, list)
        else 10
    )

    doc_gen = SynthImageGenerator(
        font_config=font_config,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        margin=margin,
    )

    min_text = (
        config.get("min_text_samples", 15)
        if not isinstance(config, list)
        else 15
    )
    max_text = (
        config.get("max_text_samples", 30)
        if not isinstance(config, list)
        else 30
    )

    pipeline = GenerationPipeline(
        source=source,
        document_generator=doc_gen,
        transform_pipeline=transform_pipeline,
        output_dir=output,
        min_text_samples=min_text,
        max_text_samples=max_text,
    )

    click.echo("Running pipeline...")
    pipeline.run(num_samples)
    click.echo("Pipeline finished.")

    # Force exit to prevent hanging on background threads
    # (common with 'datasets'/'fsspec'/multiprocessing)
    os._exit(0)


@cli.command()
@click.option(
    "--output",
    type=click.Path(),
    default="pipeline_schema.json",
    help="Output file path for the JSON schema",
)
def pipeline_schema(output: str):
    """Generate and save the JSON schema for the full generation pipeline."""
    click.echo("Generating JSON schema for GenerationPipeline...")
    s = GenerationPipeline.model_json_schema()
    with Path(output).open("w") as f:
        json.dump(s, f, indent=2)
    click.echo(f"Schema saved to {output}")


@cli.command()
@click.option(
    "--output",
    type=click.Path(),
    default="transform_pipeline_schema.json",
    help="Output file path for the JSON schema",
)
def schema(output: str):
    """Generate and save the JSON schema for the transform pipeline."""
    click.echo("Generating JSON schema for SampleTransformPipeline...")
    s = get_pipeline_schema()
    with Path(output).open("w") as f:
        json.dump(s, f, indent=2)
    click.echo(f"Schema saved to {output}")


if __name__ == "__main__":
    cli()
