import logging
import time

import openai

from rag_quality_lab.config import config
from rag_quality_lab.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)

_BATCH_SIZE = 2048
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0


class OpenAIEmbedder(BaseEmbedder):
    """Embed texts using the OpenAI embeddings API.

    Batches calls to stay under 2048 inputs per request.
    Retries on rate limit with exponential backoff (max 3 retries, starting at 1s).
    """

    def __init__(self, model: str = config.embedding_model) -> None:
        self._model = model
        self._client = openai.OpenAI(api_key=config.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start : batch_start + _BATCH_SIZE]
            results.extend(self._embed_batch(batch))

        return results

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        delay = _RETRY_BASE_DELAY
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
                # API returns embeddings in the same order as the input
                sorted_data = sorted(response.data, key=lambda d: d.index)
                return [item.embedding for item in sorted_data]
            except openai.RateLimitError as exc:
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Rate limit hit on embedding attempt %d/%d, retrying in %.1fs",
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= 2
            except openai.OpenAIError as exc:
                raise RuntimeError(
                    f"OpenAI embedding call failed (model={self._model}): {exc}"
                ) from exc

        raise RuntimeError(
            f"OpenAI embedding failed after {_MAX_RETRIES} retries due to rate limiting"
        ) from last_error
