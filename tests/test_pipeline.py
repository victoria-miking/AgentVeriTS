import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agentverits import AgentVeriTSConfig, AgentVeriTSPipeline
from agentverits.types import Interval, ScreeningResult, VisualCandidate
from test_evidence_agent import final_action


class StubScreening:
    def run(self, values):
        rows = [VisualCandidate(f"V{i}", Interval(i * 10, i * 10 + 5), 0.01) for i in range(1, 5)]
        return ScreeningResult([0.] * len(values), {"0.01": rows}, 0.01, rows)


class PipelineClient:
    def __init__(self, all_direct=False):
        self.calls = []
        self.all_direct = all_direct

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["schema_name"] == "agentverits_global_hypothesis":
            decisions = []
            for i in range(1, 5):
                action = "remove" if i == 2 else "keep"
                decisions.append({"decision_id": f"V{i}", "candidate_id": f"V{i}", "source": "candidate",
                                  "reviewed_interval": [i * 10, i * 10 + 5], "action": action,
                                  "final_interval": None if action == "remove" else [i * 10, i * 10 + 5],
                                  "confidence": 0.99 if self.all_direct or i < 3 else 0.7, "rationale": "test"})
            payload = {"normal_pattern": "periodic", "anomaly_pattern": "deviation", "missed_anomaly_scan": "none",
                       "decisions": decisions, "summary": "test"}
        elif len(self.calls) == 2:
            payload = final_action(final_action="refine", final_interval=[31, 34], additions=[
                {"discovery_id": "A1", "interval": [60, 65], "confidence": 0.97, "rationale": "new evidence"}])
        else:
            payload = final_action(final_action="remove", final_interval=None)
        return SimpleNamespace(response_id=f"resp_{len(self.calls)}", payload=payload)


class PipelineTest(unittest.TestCase):
    def config(self):
        config = AgentVeriTSConfig()
        config.render.global_width_px = 800
        config.render.global_height_px = 300
        return config

    def test_merge_matches_direct_union_verified_and_chains_across_targets(self):
        client = PipelineClient()
        with tempfile.TemporaryDirectory() as tmp:
            result = AgentVeriTSPipeline(self.config(), screening=StubScreening(), reasoning_client=client).run(
                np.sin(np.arange(700) / 20), output_dir=tmp)
            self.assertEqual(result.final_intervals, [Interval(10, 15), Interval(31, 34), Interval(60, 65)])
            self.assertEqual([c.get("previous_response_id") for c in client.calls], [None, "resp_1", "resp_2"])
            self.assertEqual(len(result.evidence_decisions), 2)
            self.assertTrue((Path(tmp) / "global_candidates.png").is_file())
            self.assertEqual(json.loads((Path(tmp) / "routing.json").read_text())["direct"], ["V1", "V2"])
            self.assertFalse((Path(tmp) / "evidence/global_rescan.json").exists())

    def test_all_high_confidence_never_calls_agent_or_rescan(self):
        client = PipelineClient(all_direct=True)
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = AgentVeriTSPipeline(self.config(), screening=StubScreening(), reasoning_client=client)
            result = pipeline.run(np.zeros(700), output_dir=tmp)
            self.assertEqual(len(client.calls), 1)
            self.assertFalse(result.evidence_decisions)
            pipeline.run(np.zeros(700), output_dir=tmp)
            self.assertNotIn("previous_response_id", client.calls[-1])


if __name__ == "__main__":
    unittest.main()
