import unittest

from agentverits.reasoning.prompts import EVIDENCE_SYSTEM_PROMPT, GLOBAL_SYSTEM_PROMPT


class PromptCleanlinessTest(unittest.TestCase):
    def test_prompts_are_agentverits_self_contained(self):
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

    def test_continuous_confidence_and_paper_modules(self):
        text = (GLOBAL_SYSTEM_PROMPT + "\n" + EVIDENCE_SYSTEM_PROMPT).lower()
        self.assertIn("[0,1]", text)
        self.assertIn("candidate assessment", text)
        self.assertIn("agentic verification", text)
        self.assertNotIn("discrete confidence", text)


if __name__ == "__main__":
    unittest.main()
