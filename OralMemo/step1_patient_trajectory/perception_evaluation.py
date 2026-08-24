"""使用 verifier LLM 对阶段1模型感知回答做事实级评估。"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from utils.json_utils import read_json, write_json
from llm_client import ChatClient

PROMPTS_DIR = Path(__file__).with_name("prompts")


def load_verifier_prompt() -> str:
    data = yaml.safe_load((PROMPTS_DIR / "perception_verifier.yaml").read_text(encoding="utf-8"))
    return data["verifier_prompt"]


def stage_evidence(evidence: dict, stage_id: str, source_turn_id: int) -> list[dict]:
    return [
        item
        for item in evidence["evidence"]
        if item.get("introduced_stage") == stage_id
        and int(item.get("source_turn_id", -1)) == source_turn_id
    ]


def metric(correct: int, total: int) -> float:
    return round(correct / total, 4) if total else 0.0


def calculate_metrics(verdict: dict, gold_evidence: list[dict]) -> dict:
    valid_ids = {item["evidence_id"] for item in gold_evidence}
    matched_ids = set(verdict.get("matched_evidence_ids", [])) & valid_ids
    claims = verdict.get("predicted_claims", [])
    predicted_count = len(claims)
    matched_claim_count = 0
    unsupported_count = 0
    hallucination_count = 0
    claim_records = []

    for claim in claims:
        claim_ids = set(claim.get("matched_evidence_ids", [])) & valid_ids
        contradiction = bool(claim.get("contradiction"))
        supported = bool(claim.get("supported")) and not contradiction and bool(claim_ids)
        hallucination = bool(claim.get("hallucination")) or contradiction
        if supported:
            matched_claim_count += 1
        else:
            unsupported_count += 1
        if hallucination:
            hallucination_count += 1
        claim_records.append({
            "claim": claim.get("claim", ""),
            "matched_evidence_ids": sorted(claim_ids),
            "supported": supported,
            "contradiction": contradiction,
            "hallucination": hallucination,
        })

    precision = metric(matched_claim_count, predicted_count)
    recall = metric(len(matched_ids), len(gold_evidence))
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    hallucination_control = metric(predicted_count - hallucination_count, predicted_count)
    return {
        "gold_evidence_count": len(gold_evidence),
        "predicted_claim_count": predicted_count,
        "matched_claim_count": matched_claim_count,
        "matched_evidence_count": len(matched_ids),
        "unsupported_or_contradictory_claim_count": unsupported_count,
        "hallucination_claim_count": hallucination_count,
        "matched_evidence_ids": sorted(matched_ids),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_control": hallucination_control,
        "claim_judgements": claim_records,
    }


def summarize_records(records: list[dict]) -> dict:
    total_gold = sum(item["metrics"]["gold_evidence_count"] for item in records)
    total_predicted = sum(item["metrics"]["predicted_claim_count"] for item in records)
    total_matched_claims = sum(item["metrics"]["matched_claim_count"] for item in records)
    total_matched_evidence = sum(item["metrics"]["matched_evidence_count"] for item in records)
    total_unsupported = sum(
        item["metrics"]["unsupported_or_contradictory_claim_count"] for item in records
    )
    total_hallucinations = sum(item["metrics"]["hallucination_claim_count"] for item in records)
    precision = metric(total_matched_claims, total_predicted)
    recall = metric(total_matched_evidence, total_gold)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return {
        "question_count": len(records),
        "gold_evidence_count": total_gold,
        "predicted_claim_count": total_predicted,
        "matched_claim_count": total_matched_claims,
        "matched_evidence_count": total_matched_evidence,
        "unsupported_or_contradictory_claim_count": total_unsupported,
        "hallucination_claim_count": total_hallucinations,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination_control": metric(total_predicted - total_hallucinations, total_predicted),
    }


class PerceptionEvaluator:
    def __init__(
        self,
        verifier: ChatClient,
        standard: dict,
        evidence: dict,
        cache_dir: Path,
        report_path: Path,
    ) -> None:
        self.verifier = verifier
        self.standard = standard
        self.evidence = evidence
        self.cache_dir = cache_dir
        self.report_path = report_path
        self.records: list[dict] = []
        self.prompt_template = load_verifier_prompt()
        self.standard_answers = {
            (stage["stage_id"], int(qa["source_turn_id"])): qa.get("assistant", "")
            for stage in standard.get("stages", [])
            for qa in stage.get("qa_pairs", [])
        }
        self.stage_types = {
            stage["stage_id"]: stage.get("stage_type") or "unknown"
            for stage in standard.get("stages", [])
        }

    def _cache_path(self, stage_id: str, source_turn_id: int) -> Path:
        return self.cache_dir / f"perception_verifier_{stage_id}_{source_turn_id}.json"

    def evaluate(self, stage: dict, qa: dict, model_answer: str) -> dict:
        stage_id = stage["stage_id"]
        source_turn_id = int(qa["source_turn_id"])
        gold_evidence = stage_evidence(self.evidence, stage_id, source_turn_id)
        evidence_payload = [
            {
                "evidence_id": item["evidence_id"],
                "fact_text": item.get("fact_text", ""),
                "normalized": item.get("normalized", {}),
            }
            for item in gold_evidence
        ]
        prompt = self.prompt_template.format(
            question=qa.get("human", "").replace("<image>", "").strip(),
            gold_answer=self.standard_answers[(stage_id, source_turn_id)],
            evidence=json.dumps(evidence_payload, ensure_ascii=False, indent=2),
            model_answer=model_answer,
        )
        cache_input = {"model": self.verifier.model, "prompt": prompt, "max_tokens": 12000}
        cache_path = self._cache_path(stage_id, source_turn_id)
        cached = read_json(cache_path) if cache_path.exists() else None
        if cached and cached.get("input") == cache_input:
            verdict = cached["result"]
        else:
            verdict = self.verifier.complete_json(prompt, max_tokens=12000)
            write_json(cache_path, {"input": cache_input, "result": verdict})
        metrics = calculate_metrics(verdict, gold_evidence)
        return {
            "stage_id": stage_id,
            "source_turn_id": source_turn_id,
            "question": qa.get("human", ""),
            "gold_answer": self.standard_answers[(stage_id, source_turn_id)],
            "model_answer": model_answer,
            "gold_evidence": gold_evidence,
            "metrics": metrics,
            "verifier_output": verdict,
        }

    def add_and_write(self, stage: dict, qa: dict, model_answer: str) -> dict:
        record = self.evaluate(stage, qa, model_answer)
        self.records = [
            item
            for item in self.records
            if (item["stage_id"], item["source_turn_id"])
            != (record["stage_id"], record["source_turn_id"])
        ]
        self.records.append(record)
        self.records.sort(key=lambda item: (item["stage_id"], item["source_turn_id"]))
        self.write_report()
        return record

    def write_report(self) -> None:
        records_by_stage_type: dict[str, list[dict]] = {}
        for record in self.records:
            stage_type = self.stage_types.get(record["stage_id"], "unknown")
            records_by_stage_type.setdefault(stage_type, []).append(record)

        report = {
            "patient_id": self.standard["patient_id"],
            "task": "stage1_perception",
            "metric_definition": {
                "precision": "matched predicted claims / predicted claims",
                "recall": "matched gold evidence / gold evidence",
                "f1": "harmonic mean of precision and recall",
                "hallucination_control": "1 - hallucination claims / predicted claims; plausible claims outside the curated evidence are not automatically hallucinations",
            },
            "overall": summarize_records(self.records),
            "by_stage_type": {
                stage_type: summarize_records(records)
                for stage_type, records in sorted(records_by_stage_type.items())
            },
            "per_question": self.records,
        }
        write_json(self.report_path, report)
