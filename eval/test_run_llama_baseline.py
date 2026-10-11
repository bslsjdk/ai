import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("run_llama_baseline.py")
SPEC = importlib.util.spec_from_file_location("run_llama_baseline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvalHarnessTests(unittest.TestCase):
    def write_jsonl(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp.name

    def test_loads_minimal_task_and_hashes_exact_bytes(self):
        path = self.write_jsonl([
            {"task_id": "T1", "category": "logic", "prompt": "Answer 2+2."}
        ])
        raw, tasks = MODULE.load_tasks(path)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "T1")
        self.assertEqual(MODULE.sha256_bytes(raw), MODULE.sha256_bytes(Path(path).read_bytes()))

    def test_rejects_scoring_material_and_unknown_fields(self):
        path = self.write_jsonl([
            {"task_id": "T1", "prompt": "Answer", "expected_answer": "42"}
        ])
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            MODULE.load_tasks(path)

    def test_rejects_duplicate_task_ids(self):
        path = self.write_jsonl([
            {"task_id": "T1", "prompt": "first"},
            {"task_id": "T1", "prompt": "second"}
        ])
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.load_tasks(path)

    def test_rejects_empty_corpus(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        with self.assertRaisesRegex(ValueError, "empty"):
            MODULE.load_tasks(tmp.name)


    def test_smoke_writes_blind_answer_and_server_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"test-model")
            result_path = root / "result.json"
            blind_path = root / "blind.jsonl"
            mapping_path = root / "mapping.json"
            old_argv = sys.argv
            original = {
                "verify_offline": MODULE.verify_offline,
                "fetch_server_properties": MODULE.fetch_server_properties,
                "request_completion": MODULE.request_completion,
                "read_proc_status": MODULE.read_proc_status,
                "read_system_memory": MODULE.read_system_memory,
                "thermal_snapshot": MODULE.thermal_snapshot,
            }
            try:
                sys.argv = [
                    "run_llama_baseline.py",
                    "--url", "http://127.0.0.1:8080",
                    "--server-pid", "123",
                    "--model-file", str(model),
                    "--out", str(result_path),
                    "--blind-out", str(blind_path),
                    "--mapping-out", str(mapping_path),
                    "--offline-confirmed",
                    "--smoke",
                ]
                MODULE.verify_offline = lambda: {"passed": True, "attempts": []}
                MODULE.fetch_server_properties = lambda url, timeout: {"available": True}
                MODULE.request_completion = lambda *args: {
                    "content": "<think>visible reasoning</think>LOCAL_INFERENCE_OK",
                    "timings": {"predicted_n": 12, "predicted_per_second": 6.5},
                }
                MODULE.read_proc_status = lambda pid: {"VmRSS_kib": 100}
                MODULE.read_system_memory = lambda: {"memavailable_kib": 1000}
                MODULE.thermal_snapshot = lambda: []
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = MODULE.main()
                self.assertEqual(exit_code, 0)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "smoke_only")
                self.assertTrue(result["smoke_expected_marker_found"])
                self.assertEqual(result["tasks"][0]["server_generated_tokens_per_second"], 6.5)
                self.assertEqual(result["tasks"][0]["process_before"]["VmRSS_kib"], 100)
                blind_rows = [json.loads(line) for line in blind_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(blind_rows), 1)
                self.assertEqual(blind_rows[0]["thinking_text"], "visible reasoning")
                self.assertEqual(blind_rows[0]["answer_text"], "LOCAL_INFERENCE_OK")
            finally:
                sys.argv = old_argv
                for name, value in original.items():
                    setattr(MODULE, name, value)

    def test_prompt_template_records_thinking_instruction(self):
        prompt = MODULE.PROMPT_TEMPLATE.format(
            prompt="TEST_MARKER", thinking_instruction="THINKING_OFF"
        )
        self.assertTrue(prompt.endswith("ANSWER:"))
        self.assertIn("TEST_MARKER", prompt)
        self.assertIn("THINKING_OFF", prompt)

    def test_separates_explicit_visible_thinking(self):
        thinking, answer = MODULE.split_visible_thinking(
            "<think>private visible trace</think>Final answer"
        )
        self.assertEqual(thinking, "private visible trace")
        self.assertEqual(answer, "Final answer")

    def test_no_think_tags_keeps_whole_output_as_answer(self):
        thinking, answer = MODULE.split_visible_thinking("Plain answer")
        self.assertEqual(thinking, "")
        self.assertEqual(answer, "Plain answer")


if __name__ == "__main__":
    unittest.main()
