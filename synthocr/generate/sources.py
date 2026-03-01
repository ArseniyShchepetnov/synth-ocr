"""Text and data sources for OCR dataset generation."""

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Annotated, Any, Literal

from datasets import load_dataset
from pydantic import BaseModel, Field, PrivateAttr


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
    languages: list[str]
    date: str = "20231101"

    _datasets: dict = PrivateAttr(default_factory=dict)
    _iterators: dict = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Initialize datasets and iterators."""
        self._datasets = {
            lang: load_dataset(
                "wikimedia/wikipedia",
                f"{self.date}.{lang}",
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
        """Iterate over all loaded languages randomly."""
        while self._iterators:
            lang = random.choice(list(self._iterators.keys()))  # noqa: S311
            try:
                item = next(self._iterators[lang])
                yield item["text"]
            except StopIteration:
                del self._iterators[lang]


SourceType = Annotated[
    WikipediaSource,
    Field(discriminator="type"),
]


class ReceiptSource(BaseModel):
    """Source for structured receipt data."""

    wiki: WikipediaSource | None = None
    item_names: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "ka": [
                "პური",
                "რძე",
                "ყველი",
                "კარაქი",
                "წყალი",
                "ღვინო",
                "ხილი",
                "ბოსტნეული",
                "ხორცი",
                "თევზი",
            ],
            "en": [
                "Bread",
                "Milk",
                "Cheese",
                "Butter",
                "Water",
                "Wine",
                "Fruit",
                "Vegetables",
                "Meat",
                "Fish",
            ],
            "ru": [
                "Хлеб",
                "Молоко",
                "Сыр",
                "Масло",
                "Вода",
                "Вино",
                "Фрукты",
                "Овощи",
                "Мясо",
                "Рыба",
            ],
        }
    )

    def generate_receipt_data(self) -> dict:
        """Generate structured receipt data."""
        num_items = random.randint(3, 12)  # noqa: S311
        items = []
        total = 0.0

        langs = list(self.item_names.keys())

        for _ in range(num_items):
            l1, l2 = random.sample(langs, 2)
            name = (
                f"{random.choice(self.item_names[l1])} / "  # noqa: S311
                f"{random.choice(self.item_names[l2])}"  # noqa: S311
            )

            sku = "".join(
                [str(random.randint(0, 9)) for _ in range(13)]  # noqa: S311
            )
            price = round(random.uniform(0.5, 100.0), 2)  # noqa: S311
            qty = random.randint(1, 5)  # noqa: S311
            line_total = round(price * qty, 2)
            total += line_total

            items.append(
                {
                    "sku": sku,
                    "name": name,
                    "price": f"{price:.2f}",
                    "qty": str(qty),
                    "total": f"{line_total:.2f}",
                }
            )

        return {
            "header": {
                "store": "MULTILINGUAL MARKET",
                "address": "Tbilisi, Georgia",
                "terminal": f"T-{random.randint(100, 999)}",  # noqa: S311
                "date": "30.01.2026 19:51:18",
            },
            "items": items,
            "total": f"{total:.2f}",
        }
