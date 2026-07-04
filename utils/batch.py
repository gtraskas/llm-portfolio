"""Optional Batch API path for the LLM rewriter (~50% cost reduction).

Mirrors pricer/batch.py. Only relevant if the LLMRewriter ablation is run
at scale. CRITICAL: results return OUT OF ORDER — always match by custom_id,
never by line position.
"""

import json

from openai import OpenAI

from utils.items import Wine
from utils.preprocessor import PREPROCESS_MODEL, SYSTEM_PROMPT, TextAssembler


class BatchRewriter:
    """Builds, submits, and reconciles Batch API rewrite jobs."""

    def __init__(self, model: str = PREPROCESS_MODEL) -> None:
        self.model = model
        self.client = OpenAI()

    def make_line(self, wine: Wine) -> str:
        """Return one Batch-API JSONL request line for a wine."""
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": wine.full},
            ],
        }
        wrapper = {
            "custom_id": str(wine.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(wrapper)

    def write_input_file(self, wines: list[Wine], filename: str) -> None:
        """Write the Batch API input JSONL."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(self.make_line(w) for w in wines))

    def submit(self, filename: str):
        """Upload the input file and create the batch job."""
        with open(filename, "rb") as f:
            batch_input = self.client.files.create(file=f, purpose="batch")
        return self.client.batches.create(
            input_file_id=batch_input.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    def reconcile(self, batch_id: str, wines: list[Wine]) -> None:
        """Fetch results and write them back — matched by custom_id ONLY."""
        batch = self.client.batches.retrieve(batch_id)
        if batch.status != "completed":
            print(f"Batch not ready: {batch.status}")
            return
        content = self.client.files.content(batch.output_file_id).text
        by_id = {str(w.id): w for w in wines}
        for line in content.splitlines():
            result = json.loads(line)
            wine = by_id.get(result["custom_id"])
            if wine is None:
                continue
            body = result["response"]["body"]
            wine.full = body["choices"][0]["message"]["content"]
            wine.summary = TextAssembler.assemble(wine)
            wine.make_prompt(wine.summary)
