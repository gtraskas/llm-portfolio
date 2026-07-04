"""Wine data-point: a tasting note with a value-for-money (VFM) target.

Target is `vfm` (0-99): log(points/price) scaled by fixed analytic bounds,
computed deterministically by utils.vfm (no fitting, no artifacts).
`points` and `price` remain as real source quantities — the frontier
zero-shot path estimates those and runs them through the same frozen scaler.
"""

from typing import Optional, Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel

PREFIX: str = "Value score: "
QUESTION: str = (
    "What is the value-for-money score of this wine, "
    "from 0 (worst value) to 99 (best value)?"
)


class Wine(BaseModel):
    """A data-point of a wine review with a value-for-money score.

    Attributes:
        title: Wine title (vintage + winery + name) — metadata, excluded
            from model input by default to avoid memorization leakage.
        points: Critic score 80-100 — source quantity for VFM.
        price: Bottle price in USD ($4-250 after curation) — source quantity.
        vfm: Value-for-money score 0-99 — THE TARGET. Set by
            utils.vfm.apply_vfm() right after curation.
        country: Country of origin.
        province: Province or state.
        region: Sub-region (region_1 in the source data).
        variety: Grape variety.
        winery: Producer name.
        full: Raw tasting-note description.
        summary: Assembled/standardized model input text.
        prompt: Fused prompt+answer string (evaluator/baseline path, week 6).
        completion: Answer-only string, paired with `prompt` for training
            frameworks that require the question/answer split (e.g. SFT
            trainers). Set by `make_training_prompt`, unrelated to `prompt`.
        id: Stable index assigned after curation.
    """

    title: str
    points: float
    price: float
    vfm: Optional[int] = None
    country: str
    province: str
    region: str
    variety: str
    winery: str
    full: Optional[str] = None
    summary: Optional[str] = None
    prompt: Optional[str] = None
    completion: Optional[str] = None
    id: Optional[int] = None

    def make_prompt(self, text: str) -> None:
        """Build the fused prompt+answer string — used by the baseline
        evaluator path. Unchanged; do not remove."""
        if self.vfm is None:
            raise ValueError("vfm is not set — run vfm.apply_vfm() first.")
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}{self.vfm}"

    def test_prompt(self) -> str:
        """Return the fused prompt with the answer stripped — for inference."""
        return self.prompt.split(PREFIX)[0] + PREFIX

    def count_tokens(self, tokenizer) -> int:
        """Token count of `summary` under the given tokenizer."""
        return len(tokenizer.encode(self.summary, add_special_tokens=False))

    def make_training_prompt(self, tokenizer, max_tokens: int, do_round: bool) -> None:
        """Build the SPLIT prompt/completion pair required by SFT trainers.

        Truncates `summary` to `max_tokens` if needed, then sets:
          - self.prompt: question + (possibly truncated) summary + answer prefix
          - self.completion: the answer only

        Separate from `make_prompt`/`prompt` above — training frameworks need
        the question and answer as two distinct fields so the loss can be
        masked to the answer only.
        """
        if self.vfm is None:
            raise ValueError("vfm is not set — run vfm.apply_vfm() first.")
        tokens = tokenizer.encode(self.summary, add_special_tokens=False)
        text = (
            tokenizer.decode(tokens[:max_tokens]).rstrip()
            if len(tokens) > max_tokens
            else self.summary
        )
        self.prompt = f"{QUESTION}\n\n{text}\n\n{PREFIX}"
        self.completion = str(round(self.vfm)) if do_round else str(self.vfm)

    def count_prompt_tokens(self, tokenizer) -> int:
        """Total token count of prompt + completion combined."""
        full = self.prompt + self.completion
        return len(tokenizer.encode(full, add_special_tokens=False))

    def to_datapoint(self) -> dict:
        """Return the {prompt, completion} record expected by SFT trainers."""
        return {"prompt": self.prompt, "completion": self.completion}

    def __repr__(self) -> str:
        return (
            f"<{self.title} = VFM {self.vfm} "
            f"({self.points:.0f} pts / ${self.price:.0f})>"
        )

    @staticmethod
    def push_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ) -> None:
        """Push Wine lists to the HuggingFace Hub as a DatasetDict."""
        DatasetDict(
            {
                "train": Dataset.from_list([w.model_dump() for w in train]),
                "validation": Dataset.from_list([w.model_dump() for w in val]),
                "test": Dataset.from_list([w.model_dump() for w in test]),
            }
        ).push_to_hub(dataset_name)

    @staticmethod
    def push_prompts_to_hub(
        dataset_name: str, train: list[Self], val: list[Self], test: list[Self]
    ) -> None:
        """Push {prompt, completion} records to the Hub — training-ready format.

        Requires `make_training_prompt` to have been called on every wine first.
        """
        DatasetDict(
            {
                "train": Dataset.from_list([w.to_datapoint() for w in train]),
                "val": Dataset.from_list([w.to_datapoint() for w in val]),
                "test": Dataset.from_list([w.to_datapoint() for w in test]),
            }
        ).push_to_hub(dataset_name)

    @classmethod
    def from_hub(cls, dataset_name: str) -> tuple[list[Self], list[Self], list[Self]]:
        """Load a curated dataset from the Hub and reconstruct Wine objects."""
        ds = load_dataset(dataset_name)
        return (
            [cls.model_validate(row) for row in ds["train"]],
            [cls.model_validate(row) for row in ds["validation"]],
            [cls.model_validate(row) for row in ds["test"]],
        )
