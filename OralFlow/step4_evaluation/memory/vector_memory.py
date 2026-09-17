from __future__ import annotations

import os
import time

import numpy as np
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI

from step4_evaluation.memory.base import MemoryMethod, format_stage_input, normalize_query


class VectorMemory(MemoryMethod):
    name = "vector_memory"

    def __init__(self, top_k: int = 8) -> None:
        super().__init__()
        self.top_k = top_k
        self._texts: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._pending = ""
        self._client = OpenAI(
            api_key=os.environ.get("EMBEDDING_OPENAI_API_KEY", "EMPTY"),
            base_url=os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=int(os.environ.get("EMBEDDING_REQUEST_TIMEOUT", "120")),
            max_retries=0,
        )
        self._model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

    def reset(self) -> None:
        self._texts = []
        self._vectors = []
        self._pending = ""

    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)

    def _embed(self, text: str) -> np.ndarray:
        for attempt in range(4):
            try:
                response = self._client.embeddings.create(model=self._model, input=text)
                break
            except (APIConnectionError, APITimeoutError, InternalServerError):
                if attempt >= 3:
                    raise
                time.sleep(2 ** attempt)
        usage = response.usage
        self.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(usage.prompt_tokens or 0) if usage else 0,
        )
        return np.asarray(response.data[0].embedding, dtype=float)

    def update(self, llm, cache_key: str) -> None:
        if not self._pending:
            return
        self._texts.append(self._pending)
        self._vectors.append(self._embed(self._pending))
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        if not self._texts:
            return ""
        query = normalize_query(query)
        if not query:
            return "\n\n".join(self._texts[-self.top_k:])
        query_vector = self._embed(query)
        scores = [
            float(np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector)))
            for vector in self._vectors
        ]
        indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:self.top_k]
        return "\n\n".join(self._texts[index] for index in indices)

    def close(self) -> None:
        self._client.close()
