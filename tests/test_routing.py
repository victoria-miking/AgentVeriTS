import unittest

from reva.reasoning.router import UncertaintyRouter
from reva.types import GlobalDecision, GlobalHypothesis, Interval


class RoutingTest(unittest.TestCase):
    def test_routing_uses_discrete_confidence_only(self):
        rows = [
            GlobalDecision("D1", "candidate", "V1", Interval(0, 2), "keep", Interval(0, 2), 1, "x"),
            GlobalDecision("D2", "candidate", "V2", Interval(10, 12), "remove", None, 2, "x"),
            GlobalDecision("D3", "candidate", "V3", Interval(20, 22), "refine", Interval(20, 23), 3, "x"),
            GlobalDecision("D4", "added", None, Interval(30, 32), "add", Interval(30, 32), 3, "x"),
        ]
        hypothesis = GlobalHypothesis("n", "a", "m", rows, "s")
        routed = UncertaintyRouter().route(hypothesis)
        self.assertEqual([x.decision_id for x in routed.uncertain], ["D1", "D2"])
        self.assertEqual([x.decision_id for x in routed.direct], ["D3", "D4"])


if __name__ == "__main__":
    unittest.main()
