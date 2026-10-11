import importlib.util
import json
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

    def test_prompt_template_keeps_task_at_end(self):
        prompt = MODULE.PROMPT_TEMPLATE.format(prompt="TEST_MARKER")
        self.assertTrue(prompt.endswith("ANSWER:"))
        self.assertIn("TEST_MARKER", prompt)


if __name__ == "__main__":
    unittest.main()
