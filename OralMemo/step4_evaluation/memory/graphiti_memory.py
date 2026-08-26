from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta, timezone
from threading import Thread

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from config import memo_api_key
from step4_evaluation.memory.base import MemoryMethod, format_stage_input, normalize_query
from utils.retry_utils import is_transient_error


class _AsyncLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coroutine, timeout: int | None = None):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(f"Graphiti operation timed out after {timeout}s") from None

    def close(self) -> None:
        async def drain_pending() -> None:
            current = asyncio.current_task()
            tasks = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
            if tasks:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await self.loop.shutdown_asyncgens()

        try:
            asyncio.run_coroutine_threadsafe(drain_pending(), self.loop).result(10)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()


class _TrackedOpenAIClient(OpenAIClient):
    def __init__(self, memory: MemoryMethod, *args, **kwargs) -> None:
        self.memory = memory
        super().__init__(*args, **kwargs)
        self.client = self.client.with_options(
            timeout=int(os.environ.get("GRAPHITI_LLM_REQUEST_TIMEOUT", "300")),
            max_retries=0,
        )

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
        parent = super()
        response = await _retry_async(lambda: parent._create_completion(*args, **kwargs))
        self._record(response)
        return response

    async def _create_structured_completion(self, *args, **kwargs):
        parent = super()
        response = await _retry_async(lambda: parent._create_structured_completion(*args, **kwargs))
        self._record(response)
        return response


class _TrackedEmbedder(OpenAIEmbedder):
    def __init__(self, memory: MemoryMethod, *args, **kwargs) -> None:
        self.memory = memory
        super().__init__(*args, **kwargs)
        self.client = self.client.with_options(
            timeout=int(os.environ.get("EMBEDDING_REQUEST_TIMEOUT", "120")),
            max_retries=0,
        )

    async def create(self, input_data):
        response = await _retry_async(
            lambda: self.client.embeddings.create(
                input=input_data,
                model=self.config.embedding_model,
            )
        )
        self.memory.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(response.usage.prompt_tokens or 0),
        )
        return response.data[0].embedding

    async def create_batch(self, input_data_list):
        response = await _retry_async(
            lambda: self.client.embeddings.create(
                input=input_data_list,
                model=self.config.embedding_model,
            )
        )
        self.memory.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(response.usage.prompt_tokens or 0),
        )
        return [item.embedding for item in response.data]


class GraphitiMemory(MemoryMethod):
    name = "graphiti_memory"

    def __init__(self, top_k: int = 8) -> None:
        super().__init__()
        self.top_k = top_k
        self._pending = ""
        self._pending_stage = ""
        self._pending_order = 0
        self._graphiti = None
        self._async_loop = None
        self._llm = None
        self._embedder = None
        self._group_id = ""
        self._request_timeout = int(os.environ.get("GRAPHITI_REQUEST_TIMEOUT", "600"))

    def setup(self, workdir, namespace: str = "") -> None:
        super().setup(workdir, namespace)
        self._group_id = re.sub(r"[^A-Za-z0-9_-]", "_", namespace)
        self._async_loop = _AsyncLoop()
        graphiti_api_key = os.environ.get("GRAPHITI_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or memo_api_key()
        llm_config = LLMConfig(
            api_key=graphiti_api_key,
            base_url=os.environ.get("GRAPHITI_OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")),
            model=os.environ.get("GRAPHITI_OPENAI_MODEL", "gpt-5-mini"),
            temperature=0,
        )
        self._llm = _TrackedOpenAIClient(self, config=llm_config, reasoning="low")
        self._embedder = _TrackedEmbedder(
            self,
            config=OpenAIEmbedderConfig(
                api_key=os.environ.get("EMBEDDING_OPENAI_API_KEY", "EMPTY"),
                base_url=os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1"),
                embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            ),
        )
        self._graphiti = Graphiti(
            uri=os.environ["GRAPHITI_NEO4J_URI"],
            user=os.environ.get("GRAPHITI_NEO4J_USER"),
            password=os.environ.get("GRAPHITI_NEO4J_PASSWORD"),
            llm_client=self._llm,
            embedder=self._embedder,
            cross_encoder=OpenAIRerankerClient(config=llm_config, client=self._llm),
        )
        self._async_loop.run(self._graphiti.build_indices_and_constraints(), self._request_timeout)

    def reset(self) -> None:
        self._pending = ""
        self._async_loop.run(clear_data(self._graphiti.driver, group_ids=[self._group_id]), self._request_timeout)

    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)
        self._pending_stage = stage["stage_id"]
        self._pending_order = int(stage["order"])

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
        ), self._request_timeout)
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        query = normalize_query(query)
        if not query:
            return ""
        edges = self._async_loop.run(self._graphiti.search(
            query=query,
            group_ids=[self._group_id],
            num_results=self.top_k,
        ), self._request_timeout)
        return "\n".join(f"- {edge.fact}" for edge in edges if edge.fact)

    def close(self) -> None:
        async def close_openai_clients() -> None:
            for holder in (self._llm, self._embedder):
                client = getattr(holder, "client", None)
                close = getattr(client, "close", None)
                if close is None:
                    continue
                result = close()
                if asyncio.iscoroutine(result):
                    await result

        if self._graphiti is not None:
            self._async_loop.run(self._graphiti.close(), self._request_timeout)
        self._async_loop.run(close_openai_clients(), self._request_timeout)
        if self._async_loop is not None:
            self._async_loop.close()


async def _retry_async(callback):
    for attempt in range(4):
        try:
            return await callback()
        except Exception as exc:
            if attempt >= 3 or not is_transient_error(exc):
                raise
            await asyncio.sleep(min(30, 2 ** attempt * 2))
    raise RuntimeError("async retry failed")
