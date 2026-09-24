import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentverits.io import load_signal_csv
from agentverits.reasoning.evidence_tools import EvidenceTools
from agentverits.types import GlobalDecision, Interval, validate_confidence
from test_reference_trace import FakeRenderer


class ToolsAndIOTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = np.arange(2500, dtype=float)
        self.decision = GlobalDecision("V1", "candidate", "V1", Interval(10, 2400), "keep", Interval(10, 2400), 0.8, "uncertain")
        self.tools = EvidenceTools(self.values, self.root, FakeRenderer(), global_image=self.root / "global.png")

    def test_raw_values_are_exact_consecutive_pages(self):
        args = {"context_points": 0, "max_points": 2048}
        first = self.tools.execute("raw", self.decision, args)
        second = self.tools.execute("raw", self.decision, dict(args, page_start=first.data["next_start"]))
        self.assertEqual(first.data["sample_indices"] + second.data["sample_indices"], list(range(10, 2401)))
        self.assertEqual(first.data["sample_values"] + second.data["sample_values"], self.values[10:2401].tolist())
        self.assertIsNone(second.data["next_start"])

    def test_tool_target_override_and_empty_baseline(self):
        obs = self.tools.execute("statistics", self.decision, {"interval": [0, 0], "baseline": "left"})
        self.assertEqual(obs.data["interval"], [0, 0])
        self.assertEqual(obs.data["baseline_count"], 0)
        self.assertIsNone(obs.data["robust_location_z"])

    def test_plot_requests_preserve_distinct_artifact_paths(self):
        a = self.tools.execute("focus", self.decision, {"context_points": 5})
        b = self.tools.execute("focus", self.decision, {"context_points": 10})
        self.assertNotEqual(a.images, b.images)

    def test_bad_tool_and_interval_parameters_are_rejected(self):
        for name, args in [("unknown", {}), ("raw", {"interval": [1.2, 2]}), ("raw", {"interval": [0, 9999]}),
                           ("focus", {"context_points": -1}), ("reference", {"count": 0})]:
            with self.subTest(name=name, args=args), self.assertRaises(ValueError):
                self.tools.execute(name, self.decision, args)

    def test_confidence_ids_and_endpoints_are_validated(self):
        for value in [float("nan"), float("inf"), -0.1, 1.1, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_confidence(value)
        with self.assertRaises(ValueError):
            GlobalDecision("../../escape", "candidate", "V1", Interval(0, 1), "keep", Interval(0, 1), 0.8, "x")
        with self.assertRaises(ValueError):
            Interval(1.5, 3)

    def test_csv_inference_never_selects_labels(self):
        path = self.root / "signal.csv"
        path.write_text("timestamp,label,sensor\n0,1,12\n1,0,13\n")
        values, _ = load_signal_csv(path)
        np.testing.assert_array_equal(values, [12, 13])
        path.write_text("label\n1\n0\n")
        with self.assertRaises(ValueError):
            load_signal_csv(path)
        path.write_text("value\nnot-a-number\nmissing\n")
        with self.assertRaises(ValueError):
            load_signal_csv(path)


if __name__ == "__main__":
    unittest.main()
