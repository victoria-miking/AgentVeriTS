import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentverits.reasoning.evidence_tools import EvidenceTools
from agentverits.types import GlobalDecision, Interval, ScaleReferenceTrace


class FakeRenderer:
    def local_plot(self, values, interval, out_path, **kwargs):
        Path(out_path).touch()
        return Path(out_path)

    def comparison_plot(self, values, query, references, out_path, **kwargs):
        Path(out_path).touch()
        self.last_query = query
        self.last_references = list(references)
        return Path(out_path)


class ReferenceTraceTest(unittest.TestCase):
    def test_reference_tool_reuses_screening_retained_references(self):
        values = np.linspace(0.0, 1.0, 1000)
        trace = ScaleReferenceTrace(
            scale=224,
            window_starts=[0, 56, 112],
            retained_reference_starts=[
                [300, 600],
                [320, 620],
                [340, 640],
            ],
        )
        decision = GlobalDecision(
            "V1", "candidate", "V1",
            Interval(100, 120), "keep", Interval(100, 120), 0.7, "uncertain",
        )
        renderer = FakeRenderer()
        with tempfile.TemporaryDirectory() as tmp:
            tools = EvidenceTools(
                values,
                tmp,
                renderer,
                global_image=Path(tmp) / "global.png",
                reference_traces={224: trace},
            )
            obs = tools.reference_context(decision, {"count": 2})
        self.assertEqual(obs.data["reference_source"], "stage1_retained_visual_references")
        self.assertEqual(obs.data["screening_query_window"], [0, 223])
        self.assertEqual(obs.data["references"], [[300, 523], [600, 823]])
        self.assertEqual(len(obs.images), 2)


if __name__ == "__main__":
    unittest.main()
