from __future__ import annotations
import os
import time

from mem0.embeddings.openai import OpenAIEmbedding

from config import memo_api_key
from step4_evaluation.memory.base import MemoryMethod, format_stage_input, normalize_query
from utils.batch_utils import log
from utils.retry_utils import is_transient_error


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
        if self.memory.send_embedding_dimensions and self._pass_dimensions_to_api:
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
            if self.memory.send_embedding_dimensions and self._pass_dimensions_to_api:
                kwargs["dimensions"] = self.config.embedding_dims
            response = self.memory._call_with_retry("embedding", lambda: self.client.embeddings.create(**kwargs))
            self.memory.add_metrics(
                embedding_calls=1,
                embedding_tokens=int(response.usage.prompt_tokens or 0),
            )
            embeddings.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return embeddings


class Mem0Memory(MemoryMethod):

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
        self.embedding_dimensions = int(os.environ.get("EMBEDDING_DIMENSIONS", "1536"))
        self.send_embedding_dimensions = os.environ.get("EMBEDDING_SEND_DIMENSIONS", "true").lower() in {"1", "true", "yes"}
        self._config_override = config
        self._pending = ""
        self._memory = None

    def setup(self, workdir, namespace: str = "") -> None:
        super().setup(workdir, namespace)
        self.user_id = namespace
        if self.storage_dir is None:
            self.storage_dir = self.workdir / "vector_store"


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
        llm_api_key = os.environ.get("MEM0_OPENAI_API_KEY") or memo_api_key()
        llm_base_url = os.environ.get("MEM0_OPENAI_BASE_URL") or os.environ.get("MEMO_OPENAI_BASE_URL", "https://api.openai.com/v1")
        llm_model = os.environ.get("MEM0_OPENAI_MODEL") or os.environ.get("MEMO_OPENAI_MODEL", "gpt-4o-mini")
        embedding_api_key = os.environ.get("EMBEDDING_OPENAI_API_KEY", "EMPTY")
        embedding_base_url = os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1")
        embedding_dims = self.embedding_dimensions

        config: dict = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "api_key": llm_api_key,
                    "openai_base_url": llm_base_url,
                    "temperature": 0.0,
                    "max_tokens": int(os.environ.get("MEM0_LLM_MAX_TOKENS", "256")),
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
                    "collection_name": "oralflow",
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
        for attempt in range(3):
            try:
                return callback()
            except Exception as exc:
                if attempt >= 2 or not is_transient_error(exc):
                    raise
                wait_seconds = min(30, 2 ** attempt * 2)
                log(
                    f"[memory][{self.namespace}][mem0/retry] {label} {type(exc).__name__}: "
                    f"{exc}; wait={wait_seconds}s next_attempt={attempt + 2}/3"
                )
                time.sleep(wait_seconds)
        raise RuntimeError(f"mem0 {label} failed")

    def reset(self) -> None:
        self._pending = ""
        self._client().delete_all(user_id=self.user_id)


    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)
        max_chars = int(os.environ.get("MEM0_MAX_STAGE_CHARS", "0"))
        if max_chars:
            self._pending = self._pending[:max_chars]

    def update(self, llm, cache_key: str) -> None:

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
