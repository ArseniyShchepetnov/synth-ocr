"""OCR dataset generation pipeline orchestration."""

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr

from synthocr.generate.image import SynthImageGenerator
from synthocr.generate.sources import SourceType
from synthocr.generate.transforms import Sample, SampleTransformPipeline


class GenerationPipeline(BaseModel):
    """Orchestrates the data generation process."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: SourceType
    document_generator: SynthImageGenerator
    transform_pipeline: SampleTransformPipeline
    output_dir: str
    min_text_samples: int = 10
    max_text_samples: int = 30

    _text_iter: Iterator[str] | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Create output directory."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_random_language(self) -> str | None:
        """Select a random language based on source configuration."""
        if not hasattr(self.source, "languages") or not self.source.languages:
            return None
        langs = [lc.code for lc in self.source.languages]
        weights = [lc.priority for lc in self.source.languages]
        return random.choices(langs, weights=weights, k=1)[0]  # noqa: S311

    def _generate_text_samples(
        self, lang: str | None, lines_needed: int, min_line_length: int = 10
    ) -> list[str]:
        """Generate a list of text samples."""
        text_samples: list[str] = []
        while len(text_samples) < lines_needed:
            try:
                if lang:
                    text = self.source.get_text(lang)
                else:
                    # Fallback to iterator if no specific lang config
                    if self._text_iter is None:
                        self._text_iter = self.source.iter_samples()
                    text = next(self._text_iter)

                raw_lines = text.replace("\n", ". ").split(". ")
                for line_raw in raw_lines:
                    clean_line = line_raw.strip()
                    if len(clean_line) > min_line_length:
                        text_samples.append(clean_line[:150])
                        if len(text_samples) >= lines_needed:
                            break
            except (StopIteration, NotImplementedError):
                break
        return text_samples

    def run(self, num_samples: int) -> None:
        """Run the generation pipeline."""
        # 1. Generate initial batch of N samples (before transforms)
        logger.debug(f"Generating {num_samples} initial samples...")
        initial_samples: list[Sample] = []
        min_line_length = 10
        for i in range(num_samples):
            lang = self._get_random_language()
            lines_needed = random.randint(  # noqa: S311
                self.min_text_samples, self.max_text_samples
            )
            text_samples = self._generate_text_samples(
                lang, lines_needed, min_line_length
            )

            if not text_samples:
                logger.debug(
                    f"Warning: Only able to generate {len(initial_samples)} "
                    "samples due to source exhaustion."
                )
                break

            sample = self.document_generator.generate(
                text_samples, language=lang
            )
            initial_samples.append(sample)
            if (i + 1) % 5 == 0:
                logger.debug(
                    f"  Generated {i + 1}/{num_samples} initial samples..."
                )

        # 2. Apply batch transformations
        logger.debug(
            f"Applying transform pipeline to {len(initial_samples)} samples..."
        )
        final_samples = self.transform_pipeline(initial_samples)

        # 3. Save final results
        logger.debug(
            f"Saving {len(final_samples)} final samples to "
            f"{self.output_dir}..."
        )
        for i, sample in enumerate(final_samples):
            self._save_sample(sample, i)
            if (i + 1) % 5 == 0:
                logger.debug(
                    f"  Saved {i + 1}/{len(final_samples)} samples..."
                )

        logger.debug("Generation complete.")

    def _save_sample(self, sample: Sample, index: int) -> None:
        """Save a single sample to disk."""
        sample_dir = Path(self.output_dir) / f"sample_{index:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        image_path = sample_dir / "image.png"
        cv2.imwrite(
            str(image_path), cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
        )

        # Save debug image with bounding boxes
        debug_image = sample.image.copy()
        for line in sample.annotation.lines:
            bbox = line.bbox.astype(np.int32)
            cv2.polylines(
                debug_image,
                [bbox],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )

        debug_path = sample_dir / "debug.png"
        cv2.imwrite(
            str(debug_path), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
        )

        # Save annotation
        anno_path = sample_dir / "annotation.json"
        anno_data = {
            "lines": [
                {"text": line.text, "bbox": line.bbox.tolist()}
                for line in sample.annotation.lines
            ]
        }
        with anno_path.open("w") as f:
            json.dump(anno_data, f, indent=2, ensure_ascii=False)
