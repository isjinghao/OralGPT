from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from openai import OpenAI, RateLimitError


class ChatClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/") + "/")
        self.model = model

    def complete_json(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8000,
                      images: list[str] | None = None, timeout: int = 300) -> dict:
        # 发起对话补全并解析为 JSON
        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
            message: dict = {"role": "user", "content": content}
        else:
            message = {"role": "user", "content": prompt}
        for attempt in range(4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[message],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                content = response.choices[0].message.content or ""
                return parse_json_object(content)
            except RateLimitError as exc:
                wait_seconds = reset_wait_seconds(str(exc))
                print(f"LLM rate limited; waiting {wait_seconds}s before retry {attempt + 1}/3.")
                time.sleep(wait_seconds)
        raise RuntimeError("LLM request failed after 4 rate-limit retries.")


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
