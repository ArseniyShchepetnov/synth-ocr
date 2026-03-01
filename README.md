# Synth-OCR

Synth-OCR is a flexible synthetic data generator for training OCR models. It orchestrates the creation of diverse document images by scattering multilingual text from Wikipedia, applying realistic structured layouts, and performing a wide array of image transformations such as rotations, noise, blurring, and rubber stamps. The generator supports English, Georgian, and Russian text, and produces both images and JSON annotations with precise bounding box coordinates.

## Installation

This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync
```

## Running the CLI

You can run the generator using the `synth-ocr` command (after `uv sync`) or directly with `uv run`.

### Generate a dataset

The `generate` command creates a batch of synthetic images and their annotations.

```bash
# Basic generation (defaults to 10 samples in English)
uv run synth-ocr generate

# Generate 50 samples mixing English and Georgian with a custom config
uv run synth-ocr generate --langs en --langs ka --num-samples 50 --config config.json

# Specify custom fonts
uv run synth-ocr generate --fonts /path/to/font.ttf --fonts /path/to/other_font.otf
```

**Options:**
- `--langs`: Languages to include (can be repeated).
- `--num-samples`: Number of samples to generate (default: 10).
- `--output`: Directory to save the results (default: `output`).
- `--config`: Path to a JSON configuration file for transforms and pipeline settings.
- `--fonts`: Paths to custom `.ttf` or `.otf` files.

### Generate Configuration Schema

You can export the JSON schema for the generation pipeline or the transform pipeline to help create valid configuration files.

```bash
# Export the full pipeline schema
uv run synth-ocr pipeline-schema --output pipeline_schema.json

# Export only the transform pipeline schema
uv run synth-ocr schema --output transform_pipeline_schema.json
```

## Project Structure

- `synthocr/generate/`: Core generation logic (images, pipelines, and data sources).
- `synthocr/generate/transforms/`: A library of image augmentation and distortion transforms.
- `tests/`: A comprehensive test suite for verifying generation and transformation logic.
