"""基于 mem0 的检索式记忆方法。

mem0(https://github.com/mem0ai/mem0) 是面向 LLM 应用的记忆层: 从输入中抽取原子化事实, 经 ADD/UPDATE/DELETE 决策写入向量库, 回答时按问题做语义检索取回相关记忆。

与本框架其它方法的映射:
  observe(stage)      -> 暂存当前阶段文本(并累积图片路径, 供多模态使用)
  update(llm, key)    -> 调 mem0.add() 写入(mem0 内部自行完成事实抽取与冲突消解; 传入的 llm 不使用)
  context(query)      -> 调 mem0.search(query) 取回 top-k 相关记忆拼成文本

"""
from __future__ import annotations
import os
from pathlib import Path

from step4_evaluation.memory.base import MemoryMethod, collect_stage_images, format_stage_input


class Mem0Memory(MemoryMethod):
    """基于 mem0 的检索增强型记忆。"""

    name = "mem0_memory"

    def __init__(
        self,
        multimodal: bool = False,
        *,
        user_id: str = "patient",
        storage_dir=None,
        search_limit: int = 8,
        embedding_model: str | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(multimodal)
        self.user_id = user_id
        self.storage_dir = storage_dir
        self.search_limit = search_limit
        self.embedding_model = embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self._config_override = config
        self._pending = ""
        self._images: list[str] = []
        self._memory = None

    def setup(self, workdir) -> None:
        # 未显式指定 storage_dir 时, 用上层分配的专属工作目录做向量库持久化(按方法/轨迹隔离)
        if self.storage_dir is None and workdir is not None:
            self.storage_dir = Path(workdir) / "vector_store"

    # mem0 接入
    def _client(self):
        if self._memory is None:
            from mem0 import Memory
            self._memory = Memory.from_config(self._config_override or self._default_config())
        return self._memory

    def _default_config(self) -> dict:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        llm_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        config: dict = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embedding_model,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                },
            },
        }
        if self.storage_dir is not None:
            config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"oralmem_{self.user_id}",
                    "path": str(self.storage_dir),
                },
            }
        return config


    def reset(self) -> None:
        self._pending = ""
        self._images = []
        self._client().delete_all(user_id=self.user_id)


    def observe(self, stage: dict) -> None:
        self._pending = format_stage_input(stage)
        for path in collect_stage_images(stage):
            if path not in self._images:
                self._images.append(path)

    def update(self, llm, cache_key: str) -> None:
        # mem0 使用自带 LLM 完成事实抽取与写入决策, 传入的 llm 不使用
        if not self._pending:
            return
        self._client().add(self._pending, user_id=self.user_id)
        self._pending = ""

    def context(self, query: str | None = None) -> str:
        client = self._client()
        if query:
            result = client.search(query, user_id=self.user_id, limit=self.search_limit)
        else:
            result = client.get_all(user_id=self.user_id)
        items = result.get("results", result) if isinstance(result, dict) else result
        lines = []
        for item in items or []:
            text = item.get("memory", "") if isinstance(item, dict) else str(item)
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def images(self) -> list[str]:
        return list(self._images)
