"""Loads spawn99/wine-reviews and applies curation + deduplication.

Mirrors pricer/loaders.py. The source dataset may ship pre-split; we
concatenate all splits, curate, dedup, and re-split ourselves with a fixed
seed so curation drops never unbalance the provided splits.
"""

from datetime import datetime

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from utils.items import Wine
from utils.parser import parse

DATASET_ID: str = "spawn99/wine-reviews"


class WineLoader:
    """Loads, curates, and deduplicates the wine-reviews dataset."""

    def __init__(self, dataset_id: str = DATASET_ID) -> None:
        self.dataset_id = dataset_id

    def load(self) -> list[Wine]:
        """Load all splits, parse with curation rules, and deduplicate.

        Returns:
            Curated, deduplicated list of Wine objects (unshuffled).
        """
        start = datetime.now()
        print(f"Loading dataset {self.dataset_id}", flush=True)
        dataset_dict = load_dataset(self.dataset_id)
        dataset = concatenate_datasets([dataset_dict[s] for s in dataset_dict])
        print(f"Raw rows: {len(dataset):,}")

        wines = [parse(row) for row in tqdm(dataset)]
        kept = [w for w in wines if w is not None]
        print(f"Survived curation: {len(kept):,}")

        deduped = self._dedup(kept)
        elapsed = (datetime.now() - start).total_seconds() / 60
        print(f"After dedup: {len(deduped):,}  ({elapsed:.1f} min)")
        return deduped

    @staticmethod
    def _dedup(wines: list[Wine]) -> list[Wine]:
        """Drop duplicates on (title, description) — leakage guard.

        The winemag source is known to contain thousands of exact
        duplicates; duplicates across splits corrupt validation.
        """
        seen: set[tuple[str, str]] = set()
        unique: list[Wine] = []
        for wine in wines:
            key = (wine.title.lower(), wine.full.lower())
            if key not in seen:
                seen.add(key)
                unique.append(wine)
        return unique
