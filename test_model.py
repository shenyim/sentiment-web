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
        self.assertIn("active_labels", result)
        self.assertIn("ehi", result)
        self.assertEqual(len(result["probs"]), 7)
        self.assertGreaterEqual(len(result["active_labels"]), 1)

    def test_positive_entry_has_positive_polarity(self):
        result = self.analyzer.analyze("I feel supported, hopeful, and happy.")
        self.assertEqual(result["polarity_label"], "POSITIVE")
        self.assertGreaterEqual(result["ehi"], 50)

    def test_negative_distress_entry_sets_high_risk(self):
        result = self.analyzer.analyze("I feel overwhelmed and I want to disappear.")
        self.assertEqual(result["risk_level"], "high")
        self.assertIn("not a crisis service", result["support_message"])

    def test_mixed_entry_can_return_multiple_active_labels(self):
        result = self.analyzer.analyze("I feel stressed, sad, and hopeful at the same time.")
        labels = {item["label"] for item in result["active_labels"]}
        self.assertGreaterEqual(len(labels), 2)
        self.assertIn("sadness", labels)
        self.assertIn("fear", labels)


if __name__ == "__main__":
    unittest.main()
