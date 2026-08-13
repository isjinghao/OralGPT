from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from threading import Semaphore
from unittest.mock import patch

from step2_evidence.evidence import extract_all_evidence
from step3_tasks.llm_tasks import generate_rubric
from step3_tasks.run_step3 import build_evaluation_tasks, run_patient
from step3_tasks.selectors import EvidenceIndex
from step4_evaluation.evaluator import run_streaming
from step4_evaluation.report import score_method
from step4_evaluation.run_step4 import evaluate_trajectory


class Client:
    model = "test-model"
    base_url = "http://test/v1"

    def log(self, *_args) -> None:
        return None


class Method:
    name = "test-memory"
    multimodal = False

    def setup(self, workdir: Path) -> None:
        workdir.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        return None

    def observe(self, _stage: dict) -> None:
        return None

    def update(self, _llm, cache_key: str) -> None:
        return None


class Settings:
    pass


def elapsed(call) -> float:
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


def method_report(name: str) -> dict:
    return {
        "method": name,
        "acc": {"overall": {"correct": 0, "total": 0, "score": 0.0}, "by_task_type": {}, "by_modality": {}},
        "ers": {"overall": {"covered": 0, "total": 0, "score": 0.0}, "by_task_type": {}, "by_modality": {}},
        "tps": {"overall_percent": None, "per_task": []},
        "followup": {"overall_percent": None, "per_task": []},
        "per_task": [],
    }


class ParallelExecutionTests(unittest.TestCase):
    def test_step2_stages_run_in_parallel_and_keep_order(self) -> None:
        stages = {
            "patient_id": "patient",
            "stages": [
                {"stage_id": f"S{i}", "modality": [], "qa_pairs": []}
                for i in range(4)
            ],
        }

        def fake_extract(_client, _patient_id, stage, _template):
            time.sleep(0.1)
            return [{
                "evidence_id": stage["stage_id"],
                "source_turn_id": "turn",
                "introduced_stage": stage["stage_id"],
                "modality": [],
                "fact_text": "fact",
                "fact_type": "other",
                "clinical_dimension": "other",
                "normalized": {},
            }]

        with patch("step2_evidence.evidence.extract_stage_evidence", side_effect=fake_extract):
            serial = elapsed(lambda: extract_all_evidence(Client(), stages, stage_workers=1))
            result = None

            def parallel_call():
                nonlocal result
                result = extract_all_evidence(Client(), stages, stage_workers=4)

            parallel = elapsed(parallel_call)

        self.assertLess(parallel, serial * 0.6)
        self.assertEqual([f"S{i}" for i in range(4)], [item["evidence_id"] for item in result["evidence"]])

    def test_step3_evidence_selection_runs_in_parallel_and_keeps_order(self) -> None:
        standard = {
            "patient_id": "patient",
            "stages": [{
                "stage_id": "S1",
                "stage_type": "treatment",
                "order": 1,
                "qa_pairs": [
                    {
                        "role": "evaluation",
                        "ask_after_stage": "S1",
                        "release_after_stage": "S1",
                        "human": f"q{i}",
                        "assistant": "a",
                    }
                    for i in range(4)
                ],
            }],
        }
        index = EvidenceIndex([], {"edges": []})

        def fake_select(*_args, **_kwargs):
            time.sleep(0.1)
            return []

        with tempfile.TemporaryDirectory() as tmp, patch(
            "step3_tasks.run_step3.select_evaluation_evidence", side_effect=fake_select
        ):
            serial = elapsed(lambda: build_evaluation_tasks(Client(), standard, index, Path(tmp), Path(tmp), 1))
            tasks = None

            def parallel_call():
                nonlocal tasks
                tasks = build_evaluation_tasks(Client(), standard, index, Path(tmp), Path(tmp), 4)

            parallel = elapsed(parallel_call)

        self.assertLess(parallel, serial * 0.6)
        self.assertEqual(
            [f"patient_treatment_{i:03d}" for i in range(1, 5)],
            [task["task_id"] for task in tasks],
        )

    def test_step3_rubrics_run_in_parallel(self) -> None:
        def prepare(root: Path) -> None:
            (root / "trajectories").mkdir(parents=True)
            (root / "evidence").mkdir()
            (root / "graph").mkdir()
            (root / "trajectories" / "standard_trajectory.json").write_text(
                json.dumps({"patient_id": "patient", "stages": []}), encoding="utf-8"
            )
            (root / "evidence" / "evidence.json").write_text(
                json.dumps({"evidence": []}), encoding="utf-8"
            )
            (root / "graph" / "evidence_graph.json").write_text(
                json.dumps({"edges": []}), encoding="utf-8"
            )

        tasks = [
            {"task_id": f"t{i}", "task_type": "treatment", "question": "q", "gold_answer": "a"}
            for i in range(4)
        ]

        def fake_rubric(_client, task, *_args):
            time.sleep(0.1)
            return {"task_id": task["task_id"]}

        def run(workers: int) -> float:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prepare(root)
                return elapsed(lambda: run_patient(root, "patient", Settings(), task_workers=workers))

        with patch("step3_tasks.run_step3.build_client", return_value=Client()), patch(
            "step3_tasks.run_step3.build_normal_tasks", return_value=tasks
        ), patch("step3_tasks.run_step3.build_evaluation_tasks", return_value=[]), patch(
            "step3_tasks.run_step3.generate_rubric", side_effect=fake_rubric
        ):
            serial = run(1)
            parallel = run(4)

        self.assertLess(parallel, serial * 0.6)

    def test_rubric_retries_until_scores_sum_to_100(self) -> None:
        invalid = {
            "max_score": 100,
            "criteria": [
                {"name": "a", "score": 58, "description": "a"},
                {"name": "b", "score": 50, "description": "b"},
            ],
        }
        valid = {
            "max_score": 100,
            "criteria": [
                {"name": "a", "score": 50, "description": "a"},
                {"name": "b", "score": 50, "description": "b"},
            ],
        }
        task = {"task_id": "task", "task_type": "treatment", "question": "q", "gold_answer": "a"}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "step3_tasks.llm_tasks.cached_completion", side_effect=[invalid, valid]
        ) as complete:
            rubric = generate_rubric(Client(), task, Path(tmp), Path("step3_tasks/prompts"))
        self.assertEqual(100, sum(item["score"] for item in rubric["criteria"]))
        self.assertEqual(2, complete.call_count)
        self.assertIn("summed to 108", complete.call_args_list[1].args[1])

    def test_step4_same_stage_answers_run_in_parallel(self) -> None:
        trajectory = {
            "patient_id": "patient",
            "stages": [{"stage_id": "S1", "order": 1}],
        }
        tasks = {"S1": [{"task_id": f"t{i}"} for i in range(4)]}

        def fake_answer(_method, task, *_args):
            time.sleep(0.1)
            return {"task_id": task["task_id"]}

        def run(workers: int):
            return run_streaming(
                Method(), trajectory, tasks, object(), None,
                image_cache={}, answer_semaphore=Semaphore(workers), answer_workers=workers,
            )

        with patch("step4_evaluation.evaluator.answer_question", side_effect=fake_answer):
            serial = elapsed(lambda: run(1))
            records = None

            def parallel_call():
                nonlocal records
                records = run(2)

            parallel = elapsed(parallel_call)

        self.assertLess(parallel, serial * 0.75)
        self.assertEqual([f"t{i}" for i in range(4)], [record["task_id"] for record in records])

    def test_step4_scoring_runs_in_parallel(self) -> None:
        records = [
            {"task_id": f"t{i}", "task_type": "longitudinal_evidence_recall", "selected_evidence": []}
            for i in range(4)
        ]

        def fake_judge(_llm, _record):
            time.sleep(0.1)
            return {
                "correct": True,
                "reason": "ok",
                "covered_evidence_count": 0,
                "total_evidence_count": 0,
                "evidence": [],
            }

        with patch("step4_evaluation.report.judge_base", side_effect=fake_judge):
            serial = elapsed(
                lambda: score_method("method", records, {}, object(), "[test]", 1, Semaphore(1))
            )
            parallel = elapsed(
                lambda: score_method("method", records, {}, object(), "[test]", 4, Semaphore(4))
            )

        self.assertLess(parallel, serial * 0.6)

    def test_step4_scoring_skips_failed_task(self) -> None:
        records = [
            {"task_id": "failed", "task_type": "longitudinal_evidence_recall", "selected_evidence": []},
            {"task_id": "passed", "task_type": "longitudinal_evidence_recall", "selected_evidence": []},
        ]
        verdict = {
            "correct": True,
            "reason": "ok",
            "covered_evidence_count": 0,
            "total_evidence_count": 0,
            "evidence": [],
        }
        with patch(
            "step4_evaluation.report.judge_base",
            side_effect=[ValueError("LLM response message.content is empty"), verdict],
        ):
            report = score_method("method", records, {}, object(), "[test]", 2, Semaphore(2))

        self.assertEqual("failed", report["failed_tasks"][0]["task_id"])
        self.assertEqual(1, report["acc"]["overall"]["total"])
        self.assertEqual("ERROR", report["per_task"][0]["metric"])
        self.assertEqual("passed", report["per_task"][1]["task_id"])

    def test_step4_methods_run_in_parallel(self) -> None:
        trajectory = {
            "trajectory_id": "trajectory",
            "trajectory_type": "standard_trajectory",
            "patient_id": "patient",
            "stages": [],
        }

        def fake_stream(method, *_args, **_kwargs):
            time.sleep(0.1)
            return []

        def run(root: Path, workers: int) -> float:
            methods = [Method(), Method()]
            methods[0].name = "first"
            methods[1].name = "second"
            with patch("step4_evaluation.run_step4.build_methods", return_value=methods), patch(
                "step4_evaluation.run_step4.run_streaming", side_effect=fake_stream
            ), patch(
                "step4_evaluation.run_step4.score_method", side_effect=lambda name, *_args: method_report(name)
            ):
                return elapsed(lambda: evaluate_trajectory(
                    trajectory, {}, {}, Client(), Client(), root,
                    ["first", "second"], False, None, method_workers=workers,
                ))

        with tempfile.TemporaryDirectory() as serial_tmp, tempfile.TemporaryDirectory() as parallel_tmp:
            serial = run(Path(serial_tmp), 1)
            parallel = run(Path(parallel_tmp), 2)

        self.assertLess(parallel, serial * 0.75)


if __name__ == "__main__":
    unittest.main()
