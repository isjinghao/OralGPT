"""基于 mem0 的检索式记忆方法。

mem0(https://github.com/mem0ai/mem0) 是面向 LLM 应用的记忆层: 从输入中抽取原子化事实, 经 ADD/UPDATE/DELETE 决策写入向量库, 回答时按问题做语义检索取回相关记忆。

与本框架其它方法的映射:
  observe(stage)      -> 暂存当前阶段文本(并累积图片路径, 供多模态使用)
  update(llm, key)    -> 调 mem0.add() 写入(mem0 内部自行完成事实抽取与冲突消解; 传入的 llm 不使用)
  context(query)      -> 调 mem0.search(query) 取回 top-k 相关记忆拼成文本

"""
from __future__ import annotations
import os
import time

from mem0.embeddings.openai import OpenAIEmbedding

from config import memo_api_key
from step4_evaluation.memory.base import MemoryMethod, format_stage_input, normalize_query
from utils.batch_utils import log
from utils.retry_utils import is_transient_error


QWEN3_EMBEDDING_DIM = 1024
MEMORY_TIMEOUT = int(os.environ.get("MEMORY_REQUEST_TIMEOUT", "300"))
EMBEDDING_TIMEOUT = int(os.environ.get("EMBEDDING_REQUEST_TIMEOUT", "120"))


class _TrackedEmbedding(OpenAIEmbedding):
    def __init__(self, memory, config) -> None:
        self.memory = memory
        super().__init__(config)
        self.client = self.client.with_options(timeout=EMBEDDING_TIMEOUT, max_retries=0)

    def embed(self, text, memory_action=None):
        text = text.replace("\n", " ")
        kwargs = {"input": [text], "model": self.config.model, "encoding_format": "float"}
        if self._pass_dimensions_to_api and "qwen3-embedding" not in self.config.model.lower():
            kwargs["dimensions"] = self.config.embedding_dims
        response = self.memory._call_with_retry("embedding", lambda: self.client.embeddings.create(**kwargs))
        self.memory.add_metrics(
            embedding_calls=1,
            embedding_tokens=int(response.usage.prompt_tokens or 0),
        )
        return response.data[0].embedding

    def embed_batch(self, texts, memory_action="add"):
        embeddings = []
        texts = [text.replace("\n", " ") for text in texts]
        for start in range(0, len(texts), 100):
            kwargs = {
                "input": texts[start:start + 100],
                "model": self.config.model,
                "encoding_format": "float",
            }
            if self._pass_dimensions_to_api and "qwen3-embedding" not in self.config.model.lower():
                kwargs["dimensions"] = self.config.embedding_dims
            response = self.memory._call_with_retry("embedding", lambda: self.client.embeddings.create(**kwargs))
            self.memory.add_metrics(
                embedding_calls=1,
                embedding_tokens=int(response.usage.prompt_tokens or 0),
            )
            embeddings.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return embeddings


class Mem0Memory(MemoryMethod):
    """基于 mem0 的检索增强型记忆。"""

    name = "mem0_memory"

    def __init__(
        self,
        *,
        user_id: str = "patient",
        storage_dir=None,
        search_limit: int = 8,
        embedding_model: str | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.user_id = user_id
        self.storage_dir = storage_dir
        self.search_limit = search_limit
        self.embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self._config_override = config
        self._pending = ""
        self._memory = None

    def setup(self, workdir, namespace: str = "") -> None:
        super().setup(workdir, namespace)
        self.user_id = namespace
        if self.storage_dir is None:
            self.storage_dir = self.workdir / "vector_store"

    # mem0 接入
    def _client(self):
        if self._memory is None:
            from mem0 import Memory
            self._memory = Memory.from_config(self._config_override or self._default_config())
            if hasattr(self._memory.llm.client, "with_options"):
                self._memory.llm.client = self._memory.llm.client.with_options(
                    timeout=MEMORY_TIMEOUT,
                    max_retries=0,
                )
            self._memory.embedding_model = _TrackedEmbedding(
                self,
                self._memory.embedding_model.config,
            )
        return self._memory



    def _default_config(self) -> dict:
        llm_api_key = os.environ.get("MEM0_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or memo_api_key()
        llm_base_url = os.environ.get("MEM0_OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        llm_model = os.environ.get("MEM0_OPENAI_MODEL", "gpt-5-mini")
        embedding_api_key = os.environ.get("EMBEDDING_OPENAI_API_KEY", "EMPTY")
        embedding_base_url = os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1")
        embedding_dims = QWEN3_EMBEDDING_DIM if "qwen3-embedding" in self.embedding_model.lower() else 1536

        config: dict = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "api_key": llm_api_key,
                    "openai_base_url": llm_base_url,
                    "temperature": 0.0,
                    "response_callback": self._record_llm_response,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "api_key": embedding_api_key,
                    "openai_base_url": embedding_base_url,
                    "embedding_dims": embedding_dims,
                },
            },
        }
        if self.storage_dir is not None:
            config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "collection_name": "oralmem",
                    "path": str(self.storage_dir),
                    "embedding_model_dims": embedding_dims,
                },
            }
        return config

    def _record_llm_response(self, _llm, response, _params) -> None:
        usage = response.usage
        self.add_metrics(
            llm_calls=1,
            input_tokens=int(usage.prompt_tokens or 0) if usage else 0,
            output_tokens=int(usage.completion_tokens or 0) if usage else 0,
        )

    def _call_with_retry(self, label: str, callback):
        for attempt in range(4):
            try:
                return callback()
            except Exception as exc:
                if attempt >= 3 or not is_transient_error(exc):
                    raise
                wait_seconds = min(30, 2 ** attempt * 2)
                log(
                    f"[memory][{self.namespace}][mem0/retry] {label} {type(exc).__name__}: "
                    f"{exc}; wait={wait_seconds}s next_attempt={attempt + 2}/4"
                )
                time.sleep(wait_seconds)
        raise RuntimeError(f"mem0 {label} failed")

    def reset(self) -> None:
        self._pending = ""
        self._client().delete_all(user_id=self.user_id)


    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)

    def update(self, llm, cache_key: str) -> None:
        # mem0 使用自带 LLM 完成事实抽取与写入决策, 传入的 llm 不使用
        if not self._pending:
            return
        self._call_with_retry("add", lambda: self._client().add(self._pending, user_id=self.user_id))
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        client = self._client()
        filters = {"user_id": self.user_id}
        query = normalize_query(query)
        if query:
            result = self._call_with_retry(
                "search",
                lambda: client.search(query, filters=filters, top_k=self.search_limit),
            )
        else:
            result = self._call_with_retry(
                "get_all",
                lambda: client.get_all(filters=filters, top_k=self.search_limit),
            )
        items = result.get("results", result) if isinstance(result, dict) else result
        lines = []
        for item in items or []:
            text = item.get("memory", "") if isinstance(item, dict) else str(item)
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def close(self) -> None:
        if self._memory is None:
            return
        self._memory.close()
        for store_name in ("vector_store", "_telemetry_vector_store"):
            store = getattr(self._memory, store_name, None)
            client = getattr(store, "client", None)
            if client is not None:
                client.close()
