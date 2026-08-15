from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timedelta, timezone
from threading import Thread

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from step4_evaluation.memory.base import MemoryMethod, collect_stage_images, format_stage_input


class _AsyncLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


class _TrackedOpenAIClient(OpenAIClient):
    def __init__(self, memory: MemoryMethod, *args, **kwargs) -> None:
        self.memory = memory
        super().__init__(*args, **kwargs)

    def _record(self, response) -> None:
        usage = response.usage
        self.memory.add_metrics(
            llm_calls=1,
            input_tokens=int(
                getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
            ),
            output_tokens=int(
                getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
            ),
        )

    async def _create_completion(self, *args, **kwargs):
        response = await super()._create_completion(*args, **kwargs)
        self._record(response)
        return response

    async def _create_structured_completion(self, *args, **kwargs):
        response = await super()._create_structured_completion(*args, **kwargs)
        self._record(response)
        return response


class _TrackedEmbedder(OpenAIEmbedder):
    def __init__(self, memory: MemoryMethod, *args, **kwargs) -> None:
        self.memory = memory
        super().__init__(*args, **kwargs)

    async def create(self, input_data):
        response = await self.client.embeddings.create(
            input=input_data,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dim,
        )
        self.memory.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(response.usage.prompt_tokens or 0),
        )
        return response.data[0].embedding[:self.config.embedding_dim]

    async def create_batch(self, input_data_list):
        response = await self.client.embeddings.create(
            input=input_data_list,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dim,
        )
        self.memory.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(response.usage.prompt_tokens or 0),
        )
        return [item.embedding[:self.config.embedding_dim] for item in response.data]


class GraphitiMemory(MemoryMethod):
    name = "graphiti_memory"

    def __init__(self, multimodal: bool = False, top_k: int = 8) -> None:
        super().__init__(multimodal)
        self.top_k = top_k
        self._pending = ""
        self._pending_stage = ""
        self._pending_order = 0
        self._images: list[str] = []
        self._graphiti = None
        self._async_loop = None
        self._group_id = ""

    def setup(self, workdir, namespace: str = "") -> None:
        super().setup(workdir, namespace)
        self._group_id = re.sub(r"[^A-Za-z0-9_-]", "_", namespace)
        self._async_loop = _AsyncLoop()
        llm_config = LLMConfig(
            api_key=os.environ["MEMO_OPENAI_API_KEY"],
            base_url=os.environ.get("MEMO_OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("MEMO_OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        llm = _TrackedOpenAIClient(self, config=llm_config, reasoning="low")
        embedder = _TrackedEmbedder(
            self,
            config=OpenAIEmbedderConfig(
                api_key=os.environ["EMBEDDING_OPENAI_API_KEY"],
                base_url=os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1"),
                embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1536")),
            ),
        )
        self._graphiti = Graphiti(
            uri=os.environ["GRAPHITI_NEO4J_URI"],
            user=os.environ.get("GRAPHITI_NEO4J_USER"),
            password=os.environ.get("GRAPHITI_NEO4J_PASSWORD"),
            llm_client=llm,
            embedder=embedder,
            cross_encoder=OpenAIRerankerClient(config=llm_config, client=llm),
        )
        self._async_loop.run(self._graphiti.build_indices_and_constraints())

    def reset(self) -> None:
        self._pending = ""
        self._images = []
        self._async_loop.run(clear_data(self._graphiti.driver, group_ids=[self._group_id]))

    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)
        self._pending_stage = stage["stage_id"]
        self._pending_order = int(stage["order"])
        for path in collect_stage_images(stage):
            if path not in self._images:
                self._images.append(path)

    def update(self, llm, cache_key: str) -> None:
        if not self._pending:
            return
        self._async_loop.run(self._graphiti.add_episode(
                name=self._pending_stage,
                episode_body=self._pending,
                source_description="OralMemo released clinical stage",
                reference_time=datetime(2000, 1, 1, tzinfo=timezone.utc) + timedelta(days=self._pending_order),
                source=EpisodeType.text,
            group_id=self._group_id,
        ))
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        if not query:
            return ""
        edges = self._async_loop.run(self._graphiti.search(
            query=query,
            group_ids=[self._group_id],
            num_results=self.top_k,
        ))
        return "\n".join(f"- {edge.fact}" for edge in edges if edge.fact)

    def images(self) -> list[str]:
        return list(self._images)

    def close(self) -> None:
        if self._graphiti is not None:
            self._async_loop.run(self._graphiti.close())
        if self._async_loop is not None:
            self._async_loop.close()
