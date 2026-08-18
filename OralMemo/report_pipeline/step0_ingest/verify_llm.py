from __future__ import annotations

import json
from pathlib import Path

from report_pipeline.step0_ingest.timeline_llm import figures_block, tpl, load_source_text


def verify_timeline(
    client,
    raw_dir: Path,
    timeline: dict,
    captions: list[dict],
    previous_issues: list[dict] | None = None,
) -> dict:
    """Verify one timeline against its source while keeping prior decisions stable."""
    fulltext, tables_text = load_source_text(raw_dir)
    prompt = tpl("timeline_verification").substitute(
        figures_block=figures_block(captions),
        previous_issues=json.dumps(previous_issues or [], ensure_ascii=False),
        timeline_json=json.dumps(timeline, ensure_ascii=False),
        tables_text=tables_text,
        fulltext=fulltext,
    )
    return client.complete_json(prompt, temperature=0.0, max_tokens=8000)
