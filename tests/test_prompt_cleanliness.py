import unittest

from reva.reasoning.prompts import EVIDENCE_SYSTEM_PROMPT, GLOBAL_SYSTEM_PROMPT


class PromptCleanlinessTest(unittest.TestCase):
    def test_no_external_method_names(self):
        text = (GLOBAL_SYSTEM_PROMPT + "\n" + EVIDENCE_SYSTEM_PROMPT).lower()
        banned = ["vlm4ts", "agenticvlm4ts", "recora", "anomalyclaw", "anomamind", "vit4ts"]
        for term in banned:
            self.assertNotIn(term, text)

    def test_discrete_confidence_and_global_rescan_are_explicit(self):
        text = (GLOBAL_SYSTEM_PROMPT + "\n" + EVIDENCE_SYSTEM_PROMPT).lower()
        self.assertIn("1 = low confidence", text)
        self.assertIn("2 = medium confidence", text)
        self.assertIn("3 = high confidence", text)
        self.assertIn("global rescan", text)


if __name__ == "__main__":
    unittest.main()
