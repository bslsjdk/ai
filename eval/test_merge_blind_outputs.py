import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("merge_blind_outputs.py")
SPEC = importlib.util.spec_from_file_location("merge_blind_outputs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BlindMergeTests(unittest.TestCase):
    def test_merges_and_randomizes_answer_only_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            mappings = {}
            for label, rows in {
                "A": [{"blind_id": "A1", "answer_text": "A answer 1"},
                      {"blind_id": "A2", "answer_text": "A answer 2"}],
                "B": [{"blind_id": "B1", "answer_text": "B answer 1"},
                      {"blind_id": "B2", "answer_text": "B answer 2"}],
            }.items():
                blind_path = root / f"{label}.jsonl"
                blind_path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
                mapping_path = root / f"{label}.json"
                task_ids = ["T1", "T2"]
                ids = [x["blind_id"] for x in rows]
                mapping_path.write_text(json.dumps({
                    ids[0]: {"task_id": task_ids[0], "run_id": label},
                    ids[1]: {"task_id": task_ids[1], "run_id": label},
                }), encoding="utf-8")
                inputs[label] = blind_path
                mappings[label] = mapping_path

            output = root / "merged.jsonl"
            private_map = root / "private.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "merge_blind_outputs.py",
                    "--input", f"A={inputs['A']}", "--mapping", f"A={mappings['A']}",
                    "--input", f"B={inputs['B']}", "--mapping", f"B={mappings['B']}",
                    "--out", str(output), "--private-map-out", str(private_map), "--seed", "1",
                ]
                MODULE.main()
            finally:
                sys.argv = old_argv

            merged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(merged), 4)
            self.assertTrue(all(set(row) == {"task_id", "candidate_id", "answer_text"} for row in merged))
            self.assertEqual({row["task_id"] for row in merged}, {"T1", "T2"})
            for task_id in ("T1", "T2"):
                self.assertEqual(sum(row["task_id"] == task_id for row in merged), 2)
            private = json.loads(private_map.read_text(encoding="utf-8"))
            self.assertEqual(len(private["candidate_map"]), 4)
            self.assertEqual({x["source_label"] for x in private["candidate_map"].values()}, {"A", "B"})

    def test_rejects_missing_candidate_for_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.jsonl").write_text('{"blind_id":"A1","answer_text":"a"}\n', encoding="utf-8")
            (root / "a.map").write_text('{"A1":{"task_id":"T1"}}', encoding="utf-8")
            (root / "b.jsonl").write_text('{"blind_id":"B2","answer_text":"b"}\n', encoding="utf-8")
            (root / "b.map").write_text('{"B2":{"task_id":"T2"}}', encoding="utf-8")
            old_argv = sys.argv
            try:
                sys.argv = [
                    "merge_blind_outputs.py",
                    "--input", f"A={root / 'a.jsonl'}", "--mapping", f"A={root / 'a.map'}",
                    "--input", f"B={root / 'b.jsonl'}", "--mapping", f"B={root / 'b.map'}",
                    "--out", str(root / "out.jsonl"), "--private-map-out", str(root / "private.json"),
                ]
                with self.assertRaisesRegex(ValueError, "not present for every candidate"):
                    MODULE.main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
