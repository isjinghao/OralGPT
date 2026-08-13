from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_client import ChatClient
from step4_evaluation.evaluator import gather_image_urls
from step4_evaluation.run_step4 import evaluate_trajectory, trajectory_completed


class FakeClient:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")


class FakeMethod:
    def __init__(self, name: str, images: list[str] | None = None) -> None:
        self.name = name
        self.multimodal = bool(images)
        self._images = images or []

    def images(self) -> list[str]:
        return list(self._images)

    def setup(self, workdir: Path) -> None:
        Path(workdir).mkdir(parents=True, exist_ok=True)


def method_report(name: str) -> dict:
    return {
        "method": name,
        "acc": {
            "overall": {"correct": 1, "total": 1, "score": 100.0},
            "by_task_type": {},
            "by_modality": {},
        },
        "ers": {
            "overall": {"covered": 1, "total": 1, "score": 100.0},
            "by_task_type": {},
            "by_modality": {},
        },
        "tps": {"overall_percent": None, "per_task": []},
        "followup": {"overall_percent": None, "per_task": []},
        "per_task": [],
    }


def task_record() -> dict:
    return {
        "task_id": "task-1",
        "task_type": "longitudinal_evidence_recall",
        "question": "question",
        "gold_answer": "gold",
        "model_answer": "answer",
        "selected_evidence": [],
    }


class Step4OptimizationTests(unittest.TestCase):
    def test_empty_content_retries_three_times(self) -> None:
        empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
        success = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))])
        create = Mock(side_effect=[empty, empty, empty, success])
        client = object.__new__(ChatClient)
        client.model = "test-model"
        client.log_prefix = "[test]"
        client.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with patch("llm_client.time.sleep") as sleep:
            result = client.complete_text("prompt")

        self.assertEqual("answer", result)
        self.assertEqual(4, create.call_count)
        self.assertEqual(3, sleep.call_count)

    def setUp(self) -> None:
        self.trajectory = {
            "trajectory_id": "trajectory-1",
            "trajectory_type": "standard_trajectory",
            "patient_id": "group__patient",
            "stages": [],
        }
        self.tasks_by_stage = {"S1": [{"task_id": "task-1", "ask_after_stage": "S1"}]}
        self.answer_client = FakeClient("answer-model", "http://answer/v1")
        self.verifier_client = FakeClient("verifier-model", "http://verifier/v1")

    def test_image_encoding_is_reused_across_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "image.png").write_bytes(b"image")
            cache: dict[Path, str | None] = {}
            first = FakeMethod("first", ["image.png"])
            second = FakeMethod("second", ["image.png"])

            with patch(
                "step4_evaluation.evaluator.encode_image",
                return_value="data:image/png;base64,aW1hZ2U=",
            ) as encode:
                self.assertEqual(1, len(gather_image_urls(first, root, cache)))
                self.assertEqual(1, len(gather_image_urls(second, root, cache)))

            encode.assert_called_once()

    def test_method_outputs_and_scores_resume_without_reexecution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            trajectory_path = out / "trajectories" / "standard_trajectory.json"
            trajectory_path.parent.mkdir(parents=True)
            trajectory_path.write_text(json.dumps(self.trajectory), encoding="utf-8")
            with patch(
                "step4_evaluation.run_step4.build_methods",
                side_effect=lambda **_: [FakeMethod("full_context_memory")],
            ), patch(
                "step4_evaluation.run_step4.run_streaming",
                return_value=[task_record()],
            ) as run_streaming, patch(
                "step4_evaluation.run_step4.score_method",
                side_effect=lambda name, *_: method_report(name),
            ) as score:
                for _ in range(2):
                    evaluate_trajectory(
                        self.trajectory,
                        self.tasks_by_stage,
                        {},
                        self.answer_client,
                        self.verifier_client,
                        out,
                        ["full_context_memory"],
                        False,
                        None,
                    )

            self.assertEqual(1, run_streaming.call_count)
            self.assertEqual(1, score.call_count)
            self.assertTrue(
                trajectory_completed(
                    out,
                    "standard_trajectory",
                    ["full_context_memory"],
                    False,
                    "answer-model",
                )
            )

    def test_existing_answers_resume_from_scoring_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch(
                "step4_evaluation.run_step4.build_methods",
                side_effect=lambda **_: [FakeMethod("full_context_memory")],
            ), patch(
                "step4_evaluation.run_step4.run_streaming",
                return_value=[task_record()],
            ) as run_streaming, patch(
                "step4_evaluation.run_step4.score_method",
                side_effect=lambda name, *_: method_report(name),
            ) as score:
                evaluate_trajectory(
                    self.trajectory,
                    self.tasks_by_stage,
                    {},
                    self.answer_client,
                    self.verifier_client,
                    out,
                    ["full_context_memory"],
                    False,
                    None,
                )
                method_report_path = (
                    out
                    / "evaluation"
                    / "standard_trajectory"
                    / "answer-model"
                    / "full_context_memory"
                    / "text"
                    / "report.json"
                )
                method_report_path.unlink()
                evaluate_trajectory(
                    self.trajectory,
                    self.tasks_by_stage,
                    {},
                    self.answer_client,
                    self.verifier_client,
                    out,
                    ["full_context_memory"],
                    False,
                    None,
                )

            self.assertEqual(1, run_streaming.call_count)
            self.assertEqual(2, score.call_count)

    def test_failed_scoring_report_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            failed_report = method_report("full_context_memory")
            failed_report["failed_tasks"] = [{"task_id": "task-1", "error": "empty"}]
            completed_report = method_report("full_context_memory")
            completed_report["failed_tasks"] = []
            with patch(
                "step4_evaluation.run_step4.build_methods",
                side_effect=lambda **_: [FakeMethod("full_context_memory")],
            ), patch(
                "step4_evaluation.run_step4.run_streaming",
                return_value=[task_record()],
            ) as run_streaming, patch(
                "step4_evaluation.run_step4.score_method",
                side_effect=[failed_report, completed_report],
            ) as score:
                evaluate_trajectory(
                    self.trajectory, self.tasks_by_stage, {}, self.answer_client,
                    self.verifier_client, out, ["full_context_memory"], False, None,
                )
                evaluate_trajectory(
                    self.trajectory, self.tasks_by_stage, {}, self.answer_client,
                    self.verifier_client, out, ["full_context_memory"], False, None,
                )

            self.assertEqual(1, run_streaming.call_count)
            self.assertEqual(2, score.call_count)

    def test_different_services_pipeline_scoring_while_next_method_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            score_started = threading.Event()
            events: list[str] = []

            def run_streaming(method, *_args, **_kwargs):
                events.append(f"answer:{method.name}")
                if method.name == "second":
                    self.assertTrue(score_started.wait(2))
                return [task_record()]

            def score(name, *_args):
                events.append(f"score:{name}")
                if name == "first":
                    score_started.set()
                return method_report(name)

            with patch(
                "step4_evaluation.run_step4.build_methods",
                return_value=[FakeMethod("first"), FakeMethod("second")],
            ), patch(
                "step4_evaluation.run_step4.run_streaming",
                side_effect=run_streaming,
            ), patch(
                "step4_evaluation.run_step4.score_method",
                side_effect=score,
            ):
                evaluate_trajectory(
                    self.trajectory,
                    self.tasks_by_stage,
                    {},
                    self.answer_client,
                    self.verifier_client,
                    Path(tmp),
                    ["first", "second"],
                    False,
                    None,
                )

            self.assertLess(events.index("score:first"), events.index("score:second"))
            self.assertIn("answer:second", events)

    def test_same_service_keeps_answer_and_scoring_serial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events: list[str] = []
            verifier = FakeClient("verifier-model", self.answer_client.base_url)

            def run_streaming(method, *_args, **_kwargs):
                events.append(f"answer:{method.name}")
                return [task_record()]

            def score(name, *_args):
                events.append(f"score:{name}")
                return method_report(name)

            with patch(
                "step4_evaluation.run_step4.build_methods",
                return_value=[FakeMethod("first"), FakeMethod("second")],
            ), patch(
                "step4_evaluation.run_step4.run_streaming",
                side_effect=run_streaming,
            ), patch(
                "step4_evaluation.run_step4.score_method",
                side_effect=score,
            ):
                evaluate_trajectory(
                    self.trajectory,
                    self.tasks_by_stage,
                    {},
                    self.answer_client,
                    verifier,
                    Path(tmp),
                    ["first", "second"],
                    False,
                    None,
                )

            self.assertEqual(
                ["answer:first", "score:first", "answer:second", "score:second"],
                events,
            )


if __name__ == "__main__":
    unittest.main()
