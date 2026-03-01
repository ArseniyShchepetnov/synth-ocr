"""Tests for Pydantic model serialization."""

from synthocr.generate.image import SynthImageGenerator
from synthocr.generate.pipeline import GenerationPipeline
from synthocr.generate.sources import WikipediaSource
from synthocr.generate.transforms import SampleTransformPipeline


def test_pydantic_serialization() -> None:
    """Test serialization and deserialization of GenerationPipeline."""
    # Mock source
    source = WikipediaSource(languages=["en"])

    # Mock generator
    doc_gen = SynthImageGenerator(
        font_config=[
            {
                "path": "/Library/Fonts/Arial.ttf",
                "priority": 10,
                "languages": ["en"],
            }
        ],
        image_size=(512, 512),
        min_font_size=12,
        max_font_size=24,
        min_blocks=2,
        max_blocks=5,
        margin=5,
    )

    # Mock transform pipeline
    transform_pipeline = SampleTransformPipeline(transforms=[])

    # Mock pipeline
    pipeline = GenerationPipeline(
        source=source,
        document_generator=doc_gen,
        transform_pipeline=transform_pipeline,
        output_dir="test_output",
        lang_config=[{"code": "en", "probability": 1.0}],
        min_text_samples=5,
        max_text_samples=10,
    )

    # Serialize to JSON
    json_data = pipeline.model_dump_json()
    assert json_data is not None

    # Deserialize from JSON
    # We need to use TypeAdapter or similar if we want to deserialize directly.
    # But GenerationPipeline is a concrete class.
    deserialized = GenerationPipeline.model_validate_json(json_data)

    assert deserialized.output_dir == "test_output"
    assert deserialized.min_text_samples == 5
    assert deserialized.source.type == "wikipedia"
    assert isinstance(deserialized.source, WikipediaSource)
    assert deserialized.document_generator.image_size == (512, 512)


if __name__ == "__main__":
    test_pydantic_serialization()
