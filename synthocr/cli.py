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
    "--langs", multiple=True, default=["en"], help="Languages to mix"
)
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
@click.option("--fonts", multiple=True, help="Paths to .ttf or .otf fonts")
def generate(
    langs: list[str],
    output: str,
    num_samples: int,
    config_file: str | None,
    fonts: list[str],
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

    font_config = []
    if isinstance(config, list):
        transform_pipeline = SampleTransformPipeline.from_config(config)
        lang_config = None
        target_langs = list(langs)
    else:
        transforms_config = config.get("transforms", [])
        transform_pipeline = SampleTransformPipeline.from_config(
            transforms_config
        )
        lang_config = config.get("languages")
        font_config = config.get("fonts", [])
        if lang_config:
            target_langs = [lc["code"] for lc in lang_config]
        else:
            target_langs = list(langs)

    if not font_config and not fonts:
        default_paths = [
            "/System/Library/Fonts/SFGeorgian.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
        font_paths = [p for p in default_paths if Path(p).exists()]
        font_config = [
            {"path": p, "priority": 10, "languages": ["en", "ka"]}
            for p in font_paths
        ]
    elif fonts:
        # Override with command line fonts if provided
        font_config = [
            {"path": f, "priority": 10, "languages": target_langs}
            for f in fonts
        ]

    click.echo(f"Loading Wikipedia source for languages: {target_langs}")
    source = WikipediaSource(languages=target_langs)

    min_blocks = config.get("min_blocks", 4)
    max_blocks = config.get("max_blocks", 10)
    margin = config.get("margin", 10)

    doc_gen = SynthImageGenerator(
        font_config=font_config,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        margin=margin,
    )

    min_text = config.get("min_text_samples", 15)
    max_text = config.get("max_text_samples", 30)

    pipeline = GenerationPipeline(
        source=source,
        document_generator=doc_gen,
        transform_pipeline=transform_pipeline,
        output_dir=output,
        lang_config=lang_config,
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
