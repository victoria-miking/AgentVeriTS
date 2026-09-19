import unittest

from reva.reasoning.prompts import EVIDENCE_SYSTEM_PROMPT, GLOBAL_SYSTEM_PROMPT


class PromptCleanlinessTest(unittest.TestCase):
    def test_no_legacy_method_names(self):
        text = (GLOBAL_SYSTEM_PROMPT + "\n" + EVIDENCE_SYSTEM_PROMPT).lower()
        banned = ["vlm4ts", "agenticvlm4ts", "recora", "anomalyclaw", "anomamind", "vit4ts"]
        for term in banned:
            self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
