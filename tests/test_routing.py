import unittest

from reva.reasoning.router import UncertaintyRouter
from reva.types import GlobalDecision, GlobalHypothesis, Interval


class RoutingTest(unittest.TestCase):
    def test_routing_uses_confidence_only(self):
        actions = ["keep", "remove", "refine", "add"]
        rows = []
        for i, action in enumerate(actions):
            source = "added" if action == "add" else "candidate"
            final = None if action == "remove" else Interval(i * 10, i * 10 + 2)
            rows.append(GlobalDecision(
                decision_id=f"D{i}", source=source, candidate_id=None if source == "added" else f"V{i}",
                reviewed_interval=Interval(i * 10, i * 10 + 2), action=action,
                final_interval=final, confidence=0.6 if i % 2 == 0 else 0.9, rationale="x",
            ))
        hypothesis = GlobalHypothesis("n", "a", "m", rows, "s")
        routed = UncertaintyRouter(0.75).route(hypothesis)
        self.assertEqual([x.decision_id for x in routed.uncertain], ["D0", "D2"])
        self.assertEqual([x.decision_id for x in routed.direct], ["D1", "D3"])


if __name__ == "__main__":
    unittest.main()
