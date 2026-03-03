"""Text and data sources for OCR dataset generation."""

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from datasets import load_dataset
from pydantic import BaseModel, Field, PrivateAttr


class SourceLanguageConfig(BaseModel):
    """Configuration for a single language in a source."""

    code: str
    priority: float = 1.0


class TextSource(BaseModel, ABC):
    """Base class for text sources."""

    type: str

    @abstractmethod
    def iter_samples(self) -> Iterator[str]:
        """Iterate over text samples."""
        raise NotImplementedError

    def get_text(self, language: str) -> str:
        """Get a single text sample for a specific language."""
        raise NotImplementedError


class WikipediaSource(TextSource):
    """Source for text from Wikipedia."""

    type: Literal["wikipedia"] = "wikipedia"
    languages: list[SourceLanguageConfig]
    date: str = "20231101"

    _datasets: dict = PrivateAttr(default_factory=dict)
    _iterators: dict = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Initialize datasets and iterators."""
        self._datasets = {
            lang.code: load_dataset(
                "wikimedia/wikipedia",
                f"{self.date}.{lang.code}",
                split="train",
                streaming=True,
            )
            for lang in self.languages
        }
        self._iterators = {
            lang: iter(ds) for lang, ds in self._datasets.items()
        }

    def get_text(self, language: str) -> str:
        """Get text for a specific language."""
        if language not in self._iterators:
            msg = f"Language {language} not loaded in source."
            raise ValueError(msg)

        try:
            item = next(self._iterators[language])
            return item["text"]
        except StopIteration:
            self._iterators[language] = iter(self._datasets[language])
            return next(self._iterators[language])["text"]

    def iter_samples(self) -> Iterator[str]:
        """Iterate over all loaded languages randomly based on priority."""
        langs = [lang.code for lang in self.languages]
        priorities = [lang.priority for lang in self.languages]

        while self._iterators:
            # Re-filter available languages in case some are exhausted
            available_indices = [
                i for i, lang in enumerate(langs) if lang in self._iterators
            ]
            if not available_indices:
                break

            current_langs = [langs[i] for i in available_indices]
            current_priorities = [priorities[i] for i in available_indices]

            lang = random.choices(  # noqa: S311
                current_langs, weights=current_priorities, k=1
            )[0]

            try:
                item = next(self._iterators[lang])
                yield item["text"]
            except StopIteration:
                del self._iterators[lang]


SourceType = Annotated[
    WikipediaSource,
    Field(discriminator="type"),
]
