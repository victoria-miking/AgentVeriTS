import unittest
from types import SimpleNamespace

from agentverits.reasoning.candidate_assessment import CandidateAssessment
from agentverits.types import Interval, VisualCandidate


class FakeClient:
    def complete(self, **kwargs):
        return SimpleNamespace(
            response_id="r1",
            payload={
                "normal_pattern": "stable",
                "anomaly_pattern": "isolated departure",
                "missed_anomaly_scan": "one added region",
                "decisions": [
                    {
                        "decision_id": "V0001", "source": "candidate", "candidate_id": "V0001",
                        "reviewed_interval": [10, 20], "action": "keep", "final_interval": [10, 20],
                        "confidence": 0.98, "rationale": "x",
                    },
                    {
                        "decision_id": "A0001", "source": "added", "candidate_id": None,
                        "reviewed_interval": [30, 35], "action": "add", "final_interval": [30, 35],
                        "confidence": 0.8, "rationale": "y",
                    },
                ],
                "summary": "ok",
            },
        )


class GlobalContractTest(unittest.TestCase):
    def test_all_candidates_and_add(self):
        candidates = [VisualCandidate("V0001", Interval(10, 20), 0.01)]
        result = CandidateAssessment(FakeClient()).run(
            signal_id="s", signal_length=100, candidates=candidates, global_image=__file__
        )
        self.assertEqual(len(result.decisions), 2)
        self.assertEqual(result.decisions[1].action, "add")
        self.assertEqual(result.decisions[1].confidence, 0.8)

    def test_missing_candidate_is_rejected(self):
        class Missing(FakeClient):
            def complete(self, **kwargs):
                reply = super().complete(**kwargs)
                reply.payload["decisions"] = reply.payload["decisions"][1:]
                return reply
        with self.assertRaises(ValueError):
            CandidateAssessment(Missing()).run(
                signal_id="s", signal_length=100,
                candidates=[VisualCandidate("V0001", Interval(10, 20), 0.01)],
                global_image=__file__,
            )

    def test_out_of_range_confidence_is_rejected(self):
        class Fractional(FakeClient):
            def complete(self, **kwargs):
                reply = super().complete(**kwargs)
                reply.payload["decisions"][0]["confidence"] = 2.5
                return reply
        with self.assertRaises(ValueError):
            CandidateAssessment(Fractional()).run(
                signal_id="s", signal_length=100,
                candidates=[VisualCandidate("V0001", Interval(10, 20), 0.01)],
                global_image=__file__,
            )


if __name__ == "__main__":
    unittest.main()
