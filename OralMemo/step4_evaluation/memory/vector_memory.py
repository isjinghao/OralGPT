from __future__ import annotations

import os

import numpy as np
from openai import OpenAI

from step4_evaluation.memory.base import MemoryMethod, collect_stage_images, format_stage_input


class VectorMemory(MemoryMethod):
    name = "vector_memory"

    def __init__(self, multimodal: bool = False, top_k: int = 8) -> None:
        super().__init__(multimodal)
        self.top_k = top_k
        self._texts: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._pending = ""
        self._images: list[str] = []
        self._client = OpenAI(
            api_key=os.environ["EMBEDDING_OPENAI_API_KEY"],
            base_url=os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        self._model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self._dimensions = int(os.environ.get("EMBEDDING_DIM", "1536"))

    def reset(self) -> None:
        self._texts = []
        self._vectors = []
        self._pending = ""
        self._images = []

    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)
        for path in collect_stage_images(stage):
            if path not in self._images:
                self._images.append(path)

    def _embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )
        self.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(response.usage.prompt_tokens or 0),
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
        if not query:
            return "\n\n".join(self._texts[-self.top_k:])
        query_vector = self._embed(query)
        scores = [
            float(np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector)))
            for vector in self._vectors
        ]
        indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:self.top_k]
        return "\n\n".join(self._texts[index] for index in indices)

    def images(self) -> list[str]:
        return list(self._images)

    def close(self) -> None:
        self._client.close()
