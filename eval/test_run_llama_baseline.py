import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        path = self.write_jsonl([{"task_id": "T1", "category": "logic", "prompt": "Answer 2+2."}])
        raw, tasks = MODULE.load_tasks(path)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_id"], "T1")
        self.assertEqual(MODULE.sha256_bytes(raw), MODULE.sha256_bytes(Path(path).read_bytes()))

    def test_rejects_scoring_material_and_unknown_fields(self):
        path = self.write_jsonl([{"task_id": "T1", "prompt": "Answer", "expected_answer": "42"}])
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            MODULE.load_tasks(path)

    def test_rejects_duplicate_task_ids(self):
        path = self.write_jsonl([
            {"task_id": "T1", "prompt": "first"},
            {"task_id": "T1", "prompt": "second"},
        ])
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.load_tasks(path)

    def test_rejects_empty_corpus(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        with self.assertRaisesRegex(ValueError, "empty"):
            MODULE.load_tasks(tmp.name)

    def test_chat_request_uses_messages_and_model_template_kwargs(self):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 3},
                }).encode()

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode())
            return FakeResponse()

        with mock.patch.object(MODULE.urllib.request, "urlopen", side_effect=fake_urlopen):
            obj, message, reason = MODULE.request_chat(
                "http://127.0.0.1:8080", "hello", 2048, 7, 30, "off"
            )
        self.assertEqual(captured["url"], "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(captured["payload"]["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(captured["payload"]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(message["content"], "answer")
        self.assertEqual(reason, "stop")

    def test_model_default_omits_thinking_override(self):
        captured = {}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode())
            return FakeResponse()

        with mock.patch.object(MODULE.urllib.request, "urlopen", side_effect=fake_urlopen):
            MODULE.request_chat("http://127.0.0.1:8080", "hello", 2048, 1, 30, "model-default")
        self.assertNotIn("chat_template_kwargs", captured["payload"])

    def test_proc_metrics_regex_parses_realistic_procfs_lines(self):
        def fake_read_text(path, *args, **kwargs):
            if str(path).endswith("/status"):
                return "VmRSS: 123 kB\\nVmHWM: 456 kB\\n"
            if str(path).endswith("/smaps_rollup"):
                return "Rss: 120 kB\\nPss: 80 kB\\nPss_Anon: 60 kB\\n"
            raise AssertionError(f"Unexpected proc path: {path}")

        with mock.patch.object(Path, "read_text", autospec=True, side_effect=fake_read_text):
            result = MODULE.read_proc_status(123)
        self.assertEqual(result["VmRSS_kib"], 123)
        self.assertEqual(result["VmHWM_kib"], 456)
        self.assertEqual(result["pss_kib"], 80)
        self.assertEqual(result["pss_anon_kib"], 60)

    def test_process_cmdline_splits_nul_delimited_arguments(self):
        with mock.patch.object(Path, "read_bytes", return_value=b"llama-server\\0-m\\0model.gguf\\0"):
            self.assertEqual(MODULE.process_cmdline(123), ["llama-server", "-m", "model.gguf"])

    def test_server_model_path_must_match_exact_resolved_file(self):
        self.assertEqual(
            MODULE.server_model_path({"body": {"model_path": "/models/a.gguf"}}),
            "/models/a.gguf",
        )
        self.assertIsNone(MODULE.server_model_path({"body": {"model_alias": "a"}}))
        self.assertEqual(MODULE.normalize_path("/tmp/../tmp/model.gguf"),
                         MODULE.normalize_path("/tmp/model.gguf"))

    def test_output_cap_detection_distinguishes_stop_from_truncation(self):
        self.assertTrue(MODULE.is_output_cap_reached("length", 2048, 2048, 2048))
        self.assertTrue(MODULE.is_output_cap_reached(None, 2048, None, 2048))
        self.assertTrue(MODULE.is_output_cap_reached(None, None, 2048, 2048))
        self.assertFalse(MODULE.is_output_cap_reached("stop", 210, 210, 2048))
        self.assertFalse(MODULE.is_output_cap_reached(None, None, None, 2048))

    def test_separates_explicit_visible_thinking(self):
        thinking, answer = MODULE.split_visible_thinking(
            "<think>visible reasoning</think>Final answer"
        )
        self.assertEqual(thinking, "visible reasoning")
        self.assertEqual(answer, "Final answer")

    def test_no_think_tags_keeps_whole_output_as_answer(self):
        thinking, answer = MODULE.split_visible_thinking("Plain answer")
        self.assertEqual(thinking, "")
        self.assertEqual(answer, "Plain answer")

    def test_smoke_writes_answer_only_and_continuous_memory_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"test-model")
            result_path = root / "result.json"
            blind_path = root / "blind.jsonl"
            mapping_path = root / "mapping.json"
            original_argv = sys.argv
            original = {
                "verify_offline": MODULE.verify_offline,
                "fetch_server_properties": MODULE.fetch_server_properties,
                "request_chat": MODULE.request_chat,
                "read_proc_status": MODULE.read_proc_status,
                "read_system_memory": MODULE.read_system_memory,
                "thermal_snapshot": MODULE.thermal_snapshot,
                "battery_snapshot": MODULE.battery_snapshot,
                "sample_during_request": MODULE.sample_during_request,
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
                    "--offline-confirmed", "--smoke",
                ]
                MODULE.verify_offline = lambda: {"passed": True, "attempts": []}
                MODULE.fetch_server_properties = lambda url, timeout: {
                    "available": True, "body": {"model_path": str(model.resolve())}
                }
                MODULE.request_chat = lambda *args: (
                    {"usage": {"completion_tokens": 12}, "timings": {"predicted_n": 12, "predicted_per_second": 6.5}},
                    {"content": "<think>visible reasoning</think>LOCAL_INFERENCE_OK"},
                    "stop",
                )
                MODULE.read_proc_status = lambda pid: {
                    "VmRSS_kib": 100, "pss_kib": 80, "VmHWM_kib": 110
                }
                MODULE.read_system_memory = lambda: {"memavailable_kib": 1000}
                MODULE.thermal_snapshot = lambda: []
                MODULE.battery_snapshot = lambda: [{"capacity": "80", "status": "Not charging"}]
                MODULE.sample_during_request = lambda pid, stop, samples, interval=0.5: samples.append({
                    "timestamp_unix": 1, "process": {"VmRSS_kib": 120, "pss_kib": 90, "VmHWM_kib": 130},
                    "system": {"memavailable_kib": 900}, "thermal": [],
                    "battery": [{"capacity": "80", "status": "Not charging"}],
                })
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = MODULE.main()
                self.assertEqual(exit_code, 0)
                result = json.loads(result_path.read_text(encoding="utf-8"))
                self.assertEqual(result["status"], "smoke_only")
                self.assertTrue(result["smoke_expected_marker_found"])
                self.assertEqual(result["tasks"][0]["completion_tokens"], 12)
                self.assertEqual(result["instrumentation"]["max_sampled_server_pss_kib"], 90)
                self.assertEqual(result["instrumentation"]["min_sampled_system_memavailable_kib"], 900)
                blind_rows = [json.loads(line) for line in blind_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(blind_rows), 1)
                self.assertEqual(set(blind_rows[0]), {"blind_id", "answer_text"})
                self.assertEqual(blind_rows[0]["answer_text"], "LOCAL_INFERENCE_OK")
                self.assertNotIn("thinking_text", blind_rows[0])
                self.assertNotIn("output_chars", blind_rows[0])
            finally:
                sys.argv = original_argv
                for name, value in original.items():
                    setattr(MODULE, name, value)


if __name__ == "__main__":
    unittest.main()
