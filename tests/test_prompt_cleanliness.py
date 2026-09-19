import unittest

from reva.reasoning.prompts import EVIDENCE_SYSTEM_PROMPT, GLOBAL_SYSTEM_PROMPT


class PromptCleanlinessTest(unittest.TestCase):
    def test_prompts_are_reva_self_contained(self):
        text = (GLOBAL_SYSTEM_PROMPT + "\n" + EVIDENCE_SYSTEM_PROMPT).lower()
        self.assertIn("coarse-grained, purely visual screening model", text)
        self.assertIn("keep", text)
        self.assertIn("remove", text)
        self.assertIn("refine", text)
        self.assertIn("add", text)
        self.assertNotIn("previous paper", text)
        self.assertNotIn("original method", text)
        self.assertNotIn("baseline prompt", text)
        self.assertNotIn("follow another", text)

    def test_discrete_confidence_and_global_rescan_are_explicit(self):
        text = (GLOBAL_SYSTEM_PROMPT + "\n" + EVIDENCE_SYSTEM_PROMPT).lower()
        self.assertIn("1 = low confidence", text)
        self.assertIn("2 = medium confidence", text)
        self.assertIn("3 = high confidence", text)
        self.assertIn("global rescan", text)


if __name__ == "__main__":
    unittest.main()
