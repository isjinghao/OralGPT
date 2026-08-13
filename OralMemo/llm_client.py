from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from batch_utils import log


class ChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, log_prefix: str = "[llm]"):
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url + "/")
        self.model = model
        self.log_prefix = log_prefix

    def log(self, scope: str, message: str) -> None:
        log(f"{self.log_prefix}[{scope}] {message}")

    @staticmethod
    def _content_text(content, nested: bool = False) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                text for item in content
                if (text := ChatClient._content_text(item, nested=True).strip())
            )
        if isinstance(content, dict):
            if not nested:
                return json.dumps(content, ensure_ascii=False)
            parts = [
                ChatClient._content_text(content[key], nested=True).strip()
                for key in ("text", "content", "memory")
                if key in content
            ]
            if parts:
                return "\n".join(part for part in parts if part)
            return json.dumps(content, ensure_ascii=False)
        for key in ("text", "content", "memory"):
            value = getattr(content, key, None)
            if value is not None:
                return ChatClient._content_text(value, nested=True)
        return str(content)

    @staticmethod
    def _messages(prompt: str, images: list[str] | None, system_prompt: str | None) -> list[dict]:
        if images:
            content: str | list[dict] = [{"type": "text", "text": prompt}]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
        else:
            content = prompt
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def _complete(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        images: list[str] | None,
        timeout: int,
        system_prompt: str | None,
    ) -> str:
        messages = self._messages(prompt, images, system_prompt)
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                if not response.choices:
                    raise ValueError("LLM response contains no choices")
                content = self._content_text(response.choices[0].message.content).strip()
                if not content:
                    if attempt >= 3:
                        self.log("llm/error", "Empty message.content after 4 attempts")
                        raise ValueError("LLM response message.content is empty")
                    self.log("llm/retry", f"Empty message.content; next_attempt={attempt + 2}/4")
                    time.sleep(5)
                    continue
                return content
            except RateLimitError as exc:
                if attempt >= 3:
                    self.log("llm/error", "RateLimitError after 4 attempts")
                    raise
                wait_seconds = reset_wait_seconds(str(exc))
                self.log("llm/retry", f"RateLimitError; wait={wait_seconds}s next_attempt={attempt + 2}/4")
                time.sleep(wait_seconds)
            except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
                if attempt >= 3:
                    self.log("llm/error", f"{type(exc).__name__} after 4 attempts")
                    raise
                wait_seconds = min(60, 2 ** attempt * 5)
                self.log("llm/retry", f"{type(exc).__name__}; wait={wait_seconds}s next_attempt={attempt + 2}/4")
                time.sleep(wait_seconds)

    def complete_json(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8000,
                      images: list[str] | None = None, timeout: int = 300,
                      system_prompt: str | None = None) -> dict:
        content = self._complete(prompt, temperature, max_tokens, images, timeout, system_prompt)
        return parse_json_object(content)

    def complete_text(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8000,
                      images: list[str] | None = None, timeout: int = 300,
                      system_prompt: str | None = None) -> str:
        return self._complete(prompt, temperature, max_tokens, images, timeout, system_prompt).strip()


def reset_wait_seconds(text: str) -> int:
    # 计算限流后的等待秒数
    match = re.search(r"Limit resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", text)
    if not match:
        return 65
    reset_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return max(5, min(180, int((reset_at - datetime.now(timezone.utc)).total_seconds()) + 3))


def _clean_object_pairs(pairs: list[tuple[str, object]]) -> dict:
    cleaned: dict = {}
    for raw_key, value in pairs:
        key = raw_key.strip().rstrip(":：").rstrip()
        if key in cleaned:
            raise ValueError(f"Duplicate JSON key after cleaning: {key!r}")
        cleaned[key] = value
    return cleaned


def parse_json_object(text: str) -> dict:
    # 从模型输出中解析 JSON 对象，并只清理确定性的键名标点噪声。
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^" + re.escape(fence) + r"(?:json)?", "", text).strip()
        if text.endswith(fence):
            text = text[:-3].strip()
    try:
        result = json.loads(text, object_pairs_hook=_clean_object_pairs)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        result = json.loads(text[start:end + 1], object_pairs_hook=_clean_object_pairs)
    if not isinstance(result, dict):
        raise TypeError(f"Expected top-level JSON object, got {type(result).__name__}")
    return result
