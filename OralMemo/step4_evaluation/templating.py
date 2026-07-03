from __future__ import annotations

import string
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def load_template(name: str) -> string.Template:
    """加载并缓存 prompts/<name>.yaml 中的 template 字段。

    功能: 读取 YAML 提示词模板, 返回 string.Template 对象。
    输入: name - 模板文件名(不含扩展名)。
    输出: string.Template - 可用 $var 占位的模板。
    """
    data = yaml.safe_load((PROMPT_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    return string.Template(data["template"])


def render(name: str, **kwargs: str) -> str:
    """渲染模板, 用 safe_substitute 避免占位缺失时报错。"""
    return load_template(name).safe_substitute(**kwargs)
