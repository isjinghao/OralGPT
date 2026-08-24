from __future__ import annotations

import json
import os
import typing

from typing_extensions import NotRequired, Required

if not hasattr(typing, "NotRequired"):
    typing.NotRequired = NotRequired
if not hasattr(typing, "Required"):
    typing.Required = Required

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager
from openai import OpenAI

from config import memo_api_key
from step4_evaluation.memory.base import MemoryMethod, format_stage_input
from step4_evaluation.templating import render


class _UsageCallback(BaseCallbackHandler):
    def __init__(self, memory: MemoryMethod) -> None:
        self.memory = memory

    def on_llm_end(self, response, **kwargs) -> None:
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        self.memory.add_metrics(
            llm_calls=1,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
        )


class _TrackedEmbeddings(Embeddings):
    def __init__(self, memory: MemoryMethod) -> None:
        self.memory = memory
        self.model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = OpenAI(
            api_key=os.environ.get("EMBEDDING_OPENAI_API_KEY", "EMPTY"),
            base_url=os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        usage = response.usage
        self.memory.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(usage.prompt_tokens or 0) if usage else 0,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LangMemMemory(MemoryMethod):
    name = "langmem_memory"

    def __init__(self, top_k: int = 8) -> None:
        super().__init__()
        self.top_k = top_k
        self._pending = ""
        self._store = None
        self._manager = None
        self._embeddings = None

    def _build(self) -> None:
        callback = _UsageCallback(self)
        model = ChatOpenAI(
            api_key=memo_api_key(),
            base_url=os.environ.get("MEMO_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("MEMO_OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            callbacks=[callback],
        )
        self._embeddings = _TrackedEmbeddings(self)
        self._store = InMemoryStore(index={"embed": self._embeddings})
        self._manager = create_memory_store_manager(
            model,
            store=self._store,
            instructions=render("langmem_memory"),
            namespace=(self.namespace,),
            query_limit=self.top_k,
        )

    def reset(self) -> None:
        self._pending = ""
        self._store = None
        self._manager = None
        self._embeddings = None
        self._build()

    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)

    def update(self, llm, cache_key: str) -> None:
        if not self._pending:
            return
        self._manager.invoke({"messages": [{"role": "user", "content": self._pending}]})
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        items = self._manager.search(query=query or "", limit=self.top_k)
        lines = []
        for item in items:
            value = item.value.get("content", item.value) if isinstance(item.value, dict) else item.value
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            lines.append(f"- {json.dumps(value, ensure_ascii=False, default=str)}")
        return "\n".join(lines)

    def close(self) -> None:
        if self._embeddings is not None:
            self._embeddings.client.close()
