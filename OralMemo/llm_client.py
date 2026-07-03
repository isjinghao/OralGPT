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
                      images: list[str] | None = None) -> dict:
        """发起对话补全并解析为 JSON。

        功能: 调用 chat.completions(timeout=90s), 将返回文本解析为 JSON; 遇限流最多重试 4 次。注: 推理模型的思维链 token 计入 max_tokens, 故各调用点需为正文 JSON 之外预留充足预算。
        输入: prompt 提示词; temperature 采样温度; max_tokens 最大生成长度;
              images 可选的图片 URL 列表(http(s) 或 data:base64), 传入时以多模态 content 分块发送。
        输出: dict - 解析后的 JSON 对象
        """
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
                    timeout=90,
                )
                content = response.choices[0].message.content or ""
                return parse_json_object(content)
            except RateLimitError as exc:
                wait_seconds = reset_wait_seconds(str(exc))
                print(f"LLM rate limited; waiting {wait_seconds}s before retry {attempt + 1}/3.")
                time.sleep(wait_seconds)
        raise RuntimeError("LLM request failed after 4 rate-limit retries.")


def reset_wait_seconds(text: str) -> int:
    """计算限流后的等待秒数。

    功能: 从限流报错文本解析 "Limit resets at" 时间, 推算需等待的秒数。
    输入: text - 限流异常的字符串。
    输出: int - 等待秒数(约束在 5~180 之间, 解析失败默认 65)。
    """
    match = re.search(r"Limit resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC", text)
    if not match:
        return 65
    reset_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return max(5, min(180, int((reset_at - datetime.now(timezone.utc)).total_seconds()) + 3))


def parse_json_object(text: str) -> dict:
    """从模型输出中解析 JSON 对象。

    功能: 去除 ``` 代码围栏后尝试 json.loads; 失败则截取首个 { 到末个 } 再解析。
    输入: text - 模型返回的原始文本。
    输出: dict - 解析出的 JSON 对象; 无法解析时抛 JSONDecodeError。
    """
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
