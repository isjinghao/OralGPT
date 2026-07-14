from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template

import yaml

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


def extract_timeline(
    client,
    raw_dir: Path,
    figures: list[dict],
    feedback_issues: list[dict] | None = None,
) -> dict:
    # 用LLM从抽取的源文本中提取结构化时间线，可带上一轮的评审反馈以自我修正
    fulltext, tables_text = load_source_text(raw_dir)
    prompt = tpl("timeline_extraction").substitute(
        figures_block=figures_block(figures),
        feedback_block=feedback_block(feedback_issues),
        tables_text=tables_text,
        fulltext=fulltext,
    )
    return client.complete_json(prompt, temperature=0.0, max_tokens=16000)
