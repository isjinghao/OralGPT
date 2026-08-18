from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template

import yaml
from openai import InternalServerError

from report_pipeline.step0_ingest.pdf_extract import read_fulltext

_PROMPT_DIR = Path(__file__).with_name("prompts")

# 通用章节标题(文献综述/参考文献等), 用于裁掉与患者事实无关的
TAIL_HEADINGS = [
    "Discussion", "References", "Competing interests", "Conflict of interest",
    "Author contributions", "Authors' contributions", "Acknowledgements",
    "Acknowledgments", "Abbreviations", "Funding", "Declarations", "Ethics approval",
]


def tpl(name: str) -> Template:
    data = yaml.safe_load((_PROMPT_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    return Template(data["template"])


def trim_tail(fulltext: str, min_pos: int = 4000) -> str:
    # 通用地在文献综述/参考文献等章节标题前截断
    cut = len(fulltext)
    for h in TAIL_HEADINGS:
        m = re.search(r"\n\s*" + re.escape(h) + r"\s*\n", fulltext, re.IGNORECASE)
        if m and m.start() > min_pos:
            cut = min(cut, m.start())
    return fulltext[:cut]


def trim_head(fulltext: str) -> str:
    # 从Case presentation类标题开始, 去掉标题/摘要/背景/引言等非病人前言
    m = re.search(r"\n\s*(Case\s+(?:presentation|report|description))\s*\n", fulltext, re.IGNORECASE)
    if m and m.start() > 200:
        return fulltext[m.start():]
    return fulltext


def load_source_text(raw_dir: Path, max_chars: int = 16000, max_table_chars: int = 6000,
                     n_tables: int = 2) -> tuple[str, str]:
    # 读取抽取的全文与表格(MinerU 表格为 HTML), 拼成给 LLM 的源文本
    fulltext = read_fulltext(raw_dir)
    fulltext = trim_tail(trim_head(fulltext))[:max_chars]
    tables = json.loads((raw_dir / "tables.json").read_text(encoding="utf-8"))
    tables_sorted = sorted(tables, key=lambda t: -len(t.get("html", "")))
    tables_text = json.dumps(tables_sorted[:n_tables], ensure_ascii=False)[:max_table_chars]
    return fulltext, tables_text


def figures_block(figures: list[dict]) -> str:
    if not figures:
        return "(none)"
    return "\n".join(f"- {f['figure']}: {f['caption']}" for f in figures)


def feedback_block(issues: list[dict] | None) -> str:
    if not issues:
        return ""
    lines = ["===== REVIEWER FEEDBACK ON YOUR PREVIOUS EXTRACTION (fix ALL of these) ====="]
    for it in issues:
        lines.append(
            f"- [{it.get('severity','?')}] at {it.get('location','?')}: {it.get('problem','')}"
            f" | source says: {it.get('source_evidence','')}"
            f" | fix: {it.get('suggested_fix','')}"
        )
    return "\n".join(lines)


def _extraction_prompt(fulltext: str, tables_text: str, figures: list[dict]) -> str:
    return tpl("timeline_extraction").substitute(
        figures_block=figures_block(figures),
        tables_text=tables_text,
        fulltext=fulltext,
    )


def extract_timeline(client, raw_dir: Path, figures: list[dict]) -> dict:
    # 首轮从源文本中提取完整时间线；服务连续 500 时仅重试一次紧凑来源。
    fulltext, tables_text = load_source_text(raw_dir)
    try:
        return client.complete_json(
            _extraction_prompt(fulltext, tables_text, figures),
            temperature=0.0,
            max_tokens=16000,
        )
    except InternalServerError:
        client.log("step0/extract", "retrying once with compact source after InternalServerError")
        fulltext, tables_text = load_source_text(raw_dir, max_chars=12000, max_table_chars=3000)
        return client.complete_json(
            _extraction_prompt(fulltext, tables_text, figures),
            temperature=0.0,
            max_tokens=8000,
        )


def repair_timeline(
    client,
    raw_dir: Path,
    figures: list[dict],
    timeline: dict,
    issues: list[dict],
) -> dict:
    # 后续轮次只替换反馈涉及的时间点，保留其余时间点不变。
    fulltext, tables_text = load_source_text(raw_dir)
    indexed_timeline = [
        {"timepoint_index": index, **timepoint}
        for index, timepoint in enumerate(timeline["timepoints"])
    ]
    prompt = tpl("timeline_repair").substitute(
        figures_block=figures_block(figures),
        feedback_block=feedback_block(issues),
        timeline_json=json.dumps(indexed_timeline, ensure_ascii=False),
        tables_text=tables_text,
        fulltext=fulltext,
    )
    result = client.complete_json(prompt, temperature=0.0, max_tokens=8000)
    repaired = {"timepoints": list(timeline["timepoints"])}
    for patch in sorted(result["repairs"], key=lambda item: item["start_index"], reverse=True):
        repaired["timepoints"][patch["start_index"]:patch["end_index"] + 1] = patch["replacement_timepoints"]
    return repaired
