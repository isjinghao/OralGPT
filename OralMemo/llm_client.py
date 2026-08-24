from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from threading import Lock

from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError

from utils.batch_utils import log


class ChatClient:
    def __init__(self, api_key: str, base_url: str, model: str, log_prefix: str = "[llm]"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url + "/")
        self.model = model
        self.log_prefix = log_prefix
        self._usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._usage_lock = Lock()

    def log(self, scope: str, message: str) -> None:
        log(f"{self.log_prefix}[{scope}] {message}")

    def usage_snapshot(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> ChatClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
        response_format: dict | None = None,
    ) -> str:
        messages = self._messages(prompt, images, system_prompt)
        request_max_tokens = max_tokens
        max_retry_tokens = int(os.environ.get("OPENAI_EMPTY_CONTENT_RETRY_MAX_TOKENS", "8192"))
        for attempt in range(4):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": request_max_tokens,
                    "timeout": timeout,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
                usage = response.usage
                with self._usage_lock:
                    self._usage["calls"] += 1
                    self._usage["input_tokens"] += int(usage.prompt_tokens or 0) if usage else 0
                    self._usage["output_tokens"] += int(usage.completion_tokens or 0) if usage else 0
                if not response.choices:
                    raise ValueError("LLM response contains no choices")
                choice = response.choices[0]
                content = self._content_text(choice.message.content).strip()
                if not content:
                    if attempt >= 3:
                        self.log("llm/error", "Empty message.content after 4 attempts")
                        raise ValueError("LLM response message.content is empty")
                    finish_reason = getattr(choice, "finish_reason", None)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
                    request_max_tokens = min(max(request_max_tokens * 2, 4096), max_retry_tokens)
                    wait_seconds = 2 ** (attempt + 1)
                    self.log(
                        "llm/retry",
                        "Empty message.content; "
                        f"finish_reason={finish_reason} completion_tokens={completion_tokens} "
                        f"next_max_tokens={request_max_tokens} "
                        f"wait={wait_seconds}s next_attempt={attempt + 2}/4",
                    )
                    time.sleep(wait_seconds)
                    continue
                return content
            except RateLimitError as exc:
                if attempt >= 3:
                    self.log("llm/error", "RateLimitError after 4 attempts")
                    raise
                wait_seconds = reset_wait_seconds(str(exc))
                self.log("llm/retry", f"RateLimitError; wait={wait_seconds}s next_attempt={attempt + 2}/4")
                time.sleep(wait_seconds)
            except BadRequestError as exc:
                if not is_upstream_request_failed(exc):
                    raise
                if attempt >= 3:
                    self.log("llm/error", "Upstream request failed after 4 attempts")
                    raise
                wait_seconds = min(60, 2 ** attempt * 5)
                self.log("llm/retry", f"Upstream request failed; wait={wait_seconds}s next_attempt={attempt + 2}/4")
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
                      system_prompt: str | None = None, required_keys: tuple[str, ...] = (),
                      validator: Callable[[dict], None] | None = None,
                      json_schema_name: str | None = None,
                      json_schema: dict | None = None) -> dict:
        response_format = None
        if json_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema_name or "structured_response",
                    "schema": json_schema,
                    "strict": True,
                },
            }
        content = self._complete(
            prompt,
            temperature,
            max_tokens,
            images,
            timeout,
            system_prompt,
            response_format=response_format,
        )
        try:
            return parse_required_json(content, required_keys, validator)
        except (TypeError, ValueError) as exc:
            self.log("llm/retry", f"Invalid JSON; requesting one repair: {exc}")
            required = f" Required top-level keys: {', '.join(required_keys)}." if required_keys else ""
            repair_prompt = (
                "Return only a valid JSON object that preserves the information in the original response."
                f"{required}\nParsing error: {exc}\n\nOriginal response:\n{content}"
            )
            repaired = self._complete(
                repair_prompt,
                0.0,
                max_tokens,
                None,
                timeout,
                system_prompt,
                response_format=response_format,
            )
            return parse_required_json(repaired, required_keys, validator)

    def complete_text(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8000,
                      images: list[str] | None = None, timeout: int = 300,
                      system_prompt: str | None = None) -> str:
        return self._complete(prompt, temperature, max_tokens, images, timeout, system_prompt)


def build_client(settings, role: str, patient_id: str, *, log_prefix: str, model: str | None = None, base_url: str | None = None) -> ChatClient:
    cfg = settings.llm_for(role)
    return ChatClient(cfg.api_key, base_url or cfg.base_url, model or cfg.model, f"{log_prefix}[{patient_id}]")


def is_upstream_request_failed(exc: Exception) -> bool:
    return "upstream request failed" in str(exc).lower()


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
        text = text.removeprefix(fence).removeprefix("json").strip()
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


def parse_required_json(
    text: str,
    required_keys: tuple[str, ...],
    validator: Callable[[dict], None] | None = None,
) -> dict:
    result = parse_json_object(text)
    missing = [key for key in required_keys if key not in result]
    if missing:
        raise ValueError(f"Missing required top-level keys: {', '.join(missing)}")
    if validator:
        validator(result)
    return result
