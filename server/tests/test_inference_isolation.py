"""Check web request isolation without loading or downloading model weights."""

import importlib.util
import sys
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np


class FakeEngine:
    model = object()

    def __init__(self):
        self.task = None
        self.session_log = []
        self.fail_next = False

    def set_task(self, task):
        self.task = task

    def predict(self, video, audio, *, record_session=True):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Failed inference")
        time.sleep(0.005)
        if record_session:
            self.session_log.append(video.copy())
        return 0.4, [0.5, 0.5], self.task, {}, {}, {}


class InferenceIsolationTests(unittest.TestCase):
    def setUp(self):
        stub = types.ModuleType("model_wrapper")
        stub.StressInferenceEngine = FakeEngine
        spec = importlib.util.spec_from_file_location(
            "isolated_inference", Path(__file__).resolve().parents[1] / "inference.py"
        )
        self.api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"model_wrapper": stub}):
            spec.loader.exec_module(self.api)
        self.engine = FakeEngine()
        self.api._engine = self.engine
        self.video = np.zeros((16, 2, 2, 3), dtype=np.uint8)
        self.audio = np.zeros(160, dtype=np.float32)

    def test_repeated_sessions_do_not_retain_frames(self):
        for task in ["Speaking", "Interview"] * 10:
            result = self.api.predict(self.video, self.audio, task)
            self.assertEqual(result["dominantModality"], task)
        self.assertEqual(self.engine.session_log, [])

    def test_concurrent_requests_keep_their_own_task(self):
        tasks = [f"task-{i}" for i in range(24)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(
                lambda task: self.api.predict(self.video, self.audio, task), tasks
            ))
        self.assertEqual([r["dominantModality"] for r in results], tasks)
        self.assertEqual(self.engine.session_log, [])

    def test_failed_request_releases_lock_for_next_session(self):
        self.engine.fail_next = True
        with self.assertRaises(RuntimeError):
            self.api.predict(self.video, self.audio, "failed")
        result = self.api.predict(self.video, self.audio, "next")
        self.assertEqual(result["dominantModality"], "next")
        self.assertEqual(self.engine.session_log, [])


if __name__ == "__main__":
    unittest.main()
