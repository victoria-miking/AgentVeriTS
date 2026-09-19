import unittest
from types import SimpleNamespace

from reva.reasoning.evidence_agent import EvidenceAgent
from reva.reasoning.evidence_tools import ToolObservation
from reva.types import GlobalDecision, GlobalHypothesis, Interval


class FakeTools:
    def execute(self, name, decision, args):
        return ToolObservation(name, "evidence", {"z": 4.2}, [])

    def execute_for_interval(self, name, interval, args, decision_id="GLOBAL_RESCAN"):
        return ToolObservation(name, "global evidence", {"interval": interval.as_list() if interval else None}, [])


class VerifyClient:
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
                    "final_action": "keep", "final_interval": [10, 20], "confidence": 3,
                    "rationale": "evidence supports retention",
                },
            }
        return SimpleNamespace(response_id=f"r{self.calls}", payload=payload)


class RescanClient:
    def __init__(self):
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            payload = {
                "search_assessment": "possible missed interval",
                "action": {
                    "type": "call_tool",
                    "tool": "stat_features",
                    "arguments": {},
                    "target_interval": [60, 70],
                    "discoveries": [],
                },
            }
        else:
            payload = {
                "search_assessment": "confirmed missed interval",
                "action": {
                    "type": "finalize",
                    "tool": None,
                    "arguments": {},
                    "target_interval": None,
                    "discoveries": [
                        {
                            "discovery_id": "R0001",
                            "interval": [60, 70],
                            "confidence": 2,
                            "rationale": "global rescan plus statistics support a missed event",
                        }
                    ],
                },
            }
        return SimpleNamespace(response_id=f"rr{self.calls}", payload=payload)


class EvidenceAgentTest(unittest.TestCase):
    def test_tool_then_finalize(self):
        decision = GlobalDecision("V1", "candidate", "V1", Interval(10, 20), "keep", Interval(10, 20), 1, "uncertain")
        hypothesis = GlobalHypothesis("normal", "anomaly", "none", [decision], "summary", "r0")
        agent = EvidenceAgent(VerifyClient(), FakeTools(), max_tool_calls=2, signal_length=100)
        result, rid = agent.verify(decision, hypothesis, previous_response_id="r0")
        self.assertEqual(result.final_action, "keep")
        self.assertEqual(result.confidence, 3)
        self.assertEqual(len(result.evidence_log), 1)
        self.assertEqual(rid, "r2")

    def test_global_rescan_can_add_missed_anomaly(self):
        decision = GlobalDecision("V1", "candidate", "V1", Interval(10, 20), "keep", Interval(10, 20), 3, "certain")
        hypothesis = GlobalHypothesis("normal", "anomaly", "none", [decision], "summary", "r0")
        agent = EvidenceAgent(RescanClient(), FakeTools(), max_global_rescan_calls=2, signal_length=100)
        discoveries, rid = agent.global_rescan(
            hypothesis, [Interval(10, 20)], previous_response_id="r0"
        )
        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].interval, Interval(60, 70))
        self.assertEqual(discoveries[0].confidence, 2)
        self.assertEqual(rid, "rr2")


if __name__ == "__main__":
    unittest.main()
