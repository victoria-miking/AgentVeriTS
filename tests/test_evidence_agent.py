import unittest
from types import SimpleNamespace

import numpy as np

from reva.reasoning.evidence_agent import EvidenceAgent
from reva.reasoning.evidence_tools import ToolObservation
from reva.types import GlobalDecision, GlobalHypothesis, Interval


class FakeTools:
    def execute(self, name, decision, args):
        return ToolObservation(name, "evidence", {"z": 4.2}, [])


class FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            payload = {
                "unresolved_question": "is it contextual?",
                "evidence_assessment": "need stats",
                "action": {
                    "type": "call_tool", "tool": "stat_features", "arguments": {},
                    "final_action": None, "final_interval": None, "confidence": None, "rationale": "",
                },
            }
        else:
            payload = {
                "unresolved_question": "resolved",
                "evidence_assessment": "supports anomaly",
                "action": {
                    "type": "finalize", "tool": None, "arguments": {},
                    "final_action": "keep", "final_interval": [10, 20], "confidence": 0.91,
                    "rationale": "evidence supports retention",
                },
            }
        return SimpleNamespace(response_id=f"r{self.calls}", payload=payload)


class EvidenceAgentTest(unittest.TestCase):
    def test_tool_then_finalize(self):
        decision = GlobalDecision("V1", "candidate", "V1", Interval(10, 20), "keep", Interval(10, 20), 0.5, "uncertain")
        hypothesis = GlobalHypothesis("normal", "anomaly", "none", [decision], "summary", "r0")
        agent = EvidenceAgent(FakeClient(), FakeTools(), max_tool_calls=2, signal_length=100)
        result, rid = agent.verify(decision, hypothesis, previous_response_id="r0")
        self.assertEqual(result.final_action, "keep")
        self.assertEqual(len(result.evidence_log), 1)
        self.assertEqual(rid, "r2")


if __name__ == "__main__":
    unittest.main()
