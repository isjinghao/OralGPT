from __future__ import annotations

import json
from pathlib import Path

from report_pipeline.step0_ingest.timeline_llm import figures_block, tpl, load_source_text


def verify_timeline(client, raw_dir: Path, timeline: dict, captions: list[dict]) -> dict:
    """校验模型(critic): 对照源文本(全文+表格+图注)核验抽取的时间线, 返回 {passed, issues}
    依据 SOURCE 判断事实支持性、数值保真、跨时间点逻辑一致与时序
    """
    fulltext, tables_text = load_source_text(raw_dir)
    prompt = tpl("timeline_verification").substitute(
        figures_block=figures_block(captions),
        timeline_json=json.dumps(timeline, ensure_ascii=False),
        tables_text=tables_text,
        fulltext=fulltext,
    )
    return client.complete_json(prompt, temperature=0.0, max_tokens=16000)
