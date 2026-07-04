"""Builds the model input text for each Wine.

Mirrors pricer/preprocessor.py, with one deliberate difference: wine tasting
notes are professionally written, so the PRIMARY path is a deterministic
template assembly (no API cost). An OPTIONAL LLM rewriter is provided for
the ablation "does standardizing prose help?".

Whatever path is used at training time MUST be used at inference time.
"""

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

from utils.items import Wine

INCLUDE_TITLE: bool = False  # title embeds vintage+winery -> memorization risk
PREPROCESS_MODEL: str = "gpt-4.1-mini"

SYSTEM_PROMPT: str = (
    "Rewrite this wine tasting note into a concise, neutral description. "
    "Keep all sensory and structural facts (fruit, acidity, tannin, oak, "
    "finish). Remove flowery filler. Respond with the rewritten note only."
)


class TextAssembler:
    """Deterministic input assembly — the primary preprocessing path."""

    @staticmethod
    def assemble(wine: Wine) -> str:
        """Build the standardized input text from note + structured context."""
        lines = [wine.full, ""]
        if INCLUDE_TITLE:
            lines.append(f"Title: {wine.title}")
        lines.extend(
            [
                f"Variety: {wine.variety}",
                f"Country: {wine.country}",
                f"Province: {wine.province}",
                f"Region: {wine.region}",
                f"Winery: {wine.winery}",
            ]
        )
        return "\n".join(lines)

    def run(self, wines: list[Wine]) -> None:
        """Set summary and prompt for all wines."""
        for wine in tqdm(wines):
            wine.summary = self.assemble(wine)
            wine.make_prompt(wine.summary)


class LLMRewriter:
    """Optional: rewrite the tasting note with a cheap model (ablation only).

    Attributes:
        model: OpenAI-compatible model name.
        workers: Thread concurrency.
    """

    def __init__(self, model: str = PREPROCESS_MODEL, workers: int = 8) -> None:
        self.model = model
        self.workers = workers
        self.client = OpenAI()

    def rewrite(self, wine: Wine) -> None:
        """Rewrite one note, then assemble; falls back to raw on error."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": wine.full},
                ],
            )
            wine.full = response.choices[0].message.content
        except Exception as error:
            print(f"Rewrite failed for id={wine.id}: {error}")
        wine.summary = TextAssembler.assemble(wine)
        wine.make_prompt(wine.summary)

    def run(self, wines: list[Wine]) -> None:
        """Rewrite and assemble all wines concurrently."""
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(tqdm(pool.map(self.rewrite, wines), total=len(wines)))
