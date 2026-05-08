import unittest

from model import create_analyzer


class EmotionAnalyzerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = create_analyzer()

    def test_predict_returns_expected_shape(self):
        result = self.analyzer.analyze("I feel grateful and calm today.")
        self.assertIn("label_name", result)
        self.assertIn("probs", result)
        self.assertIn("ehi", result)
        self.assertEqual(len(result["probs"]), 7)

    def test_positive_entry_has_positive_polarity(self):
        result = self.analyzer.analyze("I feel supported, hopeful, and happy.")
        self.assertEqual(result["polarity_label"], "POSITIVE")
        self.assertGreaterEqual(result["ehi"], 50)

    def test_negative_distress_entry_sets_high_risk(self):
        result = self.analyzer.analyze("I feel overwhelmed and I want to disappear.")
        self.assertEqual(result["risk_level"], "high")


if __name__ == "__main__":
    unittest.main()
