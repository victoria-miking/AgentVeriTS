import unittest
from types import SimpleNamespace

from agentverits.reasoning.agentic_verification import AgenticVerification
from agentverits.reasoning.evidence_tools import ToolObservation
from agentverits.types import GlobalDecision, GlobalHypothesis, Interval


def final_action(**updates):
    action = {"type": "finalize", "tool": None, "arguments": None,
              "final_action": "keep", "final_interval": [10, 20], "confidence": 0.98,
              "rationale": "evidence supports retention", "additions": []}
    action.update(updates)
    return {"unresolved_question": "resolved", "evidence_assessment": "supports anomaly", "action": action}


class FakeTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, decision, args):
        self.calls.append((name, args))
        return ToolObservation(name, "evidence", {"z": 4.2}, [])


class ScriptedClient:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(response_id=f"r{len(self.calls)}", payload=self.replies[len(self.calls) - 1])


class AgentVerificationTest(unittest.TestCase):
    def setUp(self):
        self.decision = GlobalDecision("V1", "candidate", "V1", Interval(10, 20), "keep", Interval(10, 20), 0.7, "uncertain")
        self.hypothesis = GlobalHypothesis("normal", "anomaly", "none", [self.decision], "summary", "r0")

    def test_observation_continues_response_chain_and_can_add(self):
        request = final_action(type="call_tool", tool="statistics", arguments={"interval": [60, 70]},
                               final_action=None, final_interval=None, confidence=None)
        finish = final_action(additions=[{"discovery_id": "A1", "interval": [60, 70], "confidence": 0.96, "rationale": "evidence"}])
        client, tools = ScriptedClient([request, finish]), FakeTools()
        result, rid = AgenticVerification(client, tools, max_tool_calls=1, signal_length=100).verify(
            self.decision, self.hypothesis, previous_response_id="r0")
        self.assertEqual([c["previous_response_id"] for c in client.calls], ["r0", "r1"])
        self.assertEqual(result.additions[0].interval, Interval(60, 70))
        self.assertEqual(result.confidence, 0.98)
        self.assertEqual(rid, "r2")
        self.assertEqual(len(tools.calls), 1)
        action_schema = client.calls[-1]["schema"]["properties"]["action"]["properties"]
        self.assertEqual(action_schema["type"]["enum"], ["finalize"])

    def test_immediate_finalization_requires_no_tools(self):
        client, tools = ScriptedClient([final_action()]), FakeTools()
        result, _ = AgenticVerification(client, tools, max_tool_calls=0, signal_length=100).verify(
            self.decision, self.hypothesis, previous_response_id="r0")
        self.assertEqual(result.final_action, "keep")
        self.assertFalse(tools.calls)

    def test_keep_cannot_silently_refine_boundaries(self):
        client = ScriptedClient([final_action(final_interval=[9, 20])])
        with self.assertRaisesRegex(ValueError, "keep must preserve"):
            AgenticVerification(client, FakeTools(), signal_length=100).verify(self.decision, self.hypothesis, previous_response_id="r0")

    def test_high_confidence_and_missing_chain_are_rejected(self):
        agent = AgenticVerification(ScriptedClient([]), FakeTools(), signal_length=100)
        with self.assertRaisesRegex(ValueError, "preceding response"):
            agent.verify(self.decision, self.hypothesis, previous_response_id=None)
        self.decision.confidence = 0.95
        with self.assertRaisesRegex(ValueError, "bypass"):
            agent.verify(self.decision, self.hypothesis, previous_response_id="r0")

    def test_unknown_operation_is_not_accepted(self):
        client = ScriptedClient([final_action(final_action="whatever")])
        with self.assertRaisesRegex(ValueError, "unknown final"):
            AgenticVerification(client, FakeTools(), signal_length=100).verify(self.decision, self.hypothesis, previous_response_id="r0")


if __name__ == "__main__":
    unittest.main()
