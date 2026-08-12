from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from batch_utils import log


class ChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, log_prefix: str = "[llm]"):
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/")
        self.model = model
        self.log_prefix = log_prefix

    def log(self, scope: str, message: str) -> None:
        log(f"{self.log_prefix}[{scope}] {message}")

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
                return response.choices[0].message.content or ""
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


def parse_json_object(text: str) -> dict:
    # 从模型输出中解析 JSON 对象
    text = text.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^" + re.escape(fence) + r"(?:json)?", "", text).strip()
        if text.endswith(fence):
            text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])
