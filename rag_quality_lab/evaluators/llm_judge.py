from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import openai

from rag_quality_lab.config import config

logger = logging.getLogger(__name__)

_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "record_judge_scores",
        "description": "Record faithfulness, relevance, and answer correctness scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "faithfulness": {
                    "type": "number",
                    "description": "How faithfully the chunk supports the reference answer (0-1).",
                },
                "relevance": {
                    "type": "number",
                    "description": "How relevant the chunk is to the query (0-1).",
                },
                "answer_correctness": {
                    "type": "number",
                    "description": (
                        "How correct the chunk is as an answer to the query, "
                        "relative to the reference answer (0-1)."
                    ),
                },
            },
            "required": ["faithfulness", "relevance", "answer_correctness"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You are an expert evaluator for retrieval-augmented generation systems. "
    "Score the provided chunk on three dimensions, each from 0.0 to 1.0."
)


@dataclass
class JudgeScore:
    faithfulness: float
    relevance: float
    answer_correctness: float


class LLMJudge:
    """LLM-as-judge scorer using OpenAI function calling with result caching."""

    def __init__(
        self,
        model: str = config.judge_model,
        cache_path: Path = config.judge_cache_path,
    ) -> None:
        self._model = model
        self._cache_path = cache_path
        self._client = openai.OpenAI(api_key=config.openai_api_key)
        self._cache: dict[str, dict[str, float]] = self._load_cache()

    def score(self, query: str, chunk_text: str, reference_answer: str) -> JudgeScore:
        cache_key = self._make_cache_key(query, chunk_text, reference_answer)

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return JudgeScore(
                faithfulness=cached["faithfulness"],
                relevance=cached["relevance"],
                answer_correctness=cached["answer_correctness"],
            )

        result = self._call_judge(query, chunk_text, reference_answer)
        self._cache[cache_key] = {
            "faithfulness": result.faithfulness,
            "relevance": result.relevance,
            "answer_correctness": result.answer_correctness,
        }
        self._save_cache()
        return result

    def _call_judge(self, query: str, chunk_text: str, reference_answer: str) -> JudgeScore:
        user_message = (
            f"Query: {query}\n\n"
            f"Retrieved chunk:\n{chunk_text}\n\n"
            f"Reference answer:\n{reference_answer}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=[_TOOL_DEFINITION],
            tool_choice={"type": "function", "function": {"name": "record_judge_scores"}},
        )

        tool_call = response.choices[0].message.tool_calls
        if not tool_call:
            raise RuntimeError("LLM judge returned no tool call — cannot extract scores")

        args = json.loads(tool_call[0].function.arguments)
        return JudgeScore(
            faithfulness=float(args["faithfulness"]),
            relevance=float(args["relevance"]),
            answer_correctness=float(args["answer_correctness"]),
        )

    def _make_cache_key(self, query: str, chunk_text: str, reference_answer: str) -> str:
        raw = f"{self._model}|{query}|{chunk_text}|{reference_answer}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_cache(self) -> dict[str, dict[str, float]]:
        if self._cache_path.exists():
            try:
                return json.loads(self._cache_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load judge cache from %s: %s", self._cache_path, exc)
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2))
