import unittest
from src.services.emotion_analysis import analyze_emotion

class TestEmotionAnalysis(unittest.TestCase):
    def test_analyze_emotion(self):
        result = analyze_emotion(None)
        self.assertEqual(result["error"], "No image provided")

if __name__ == "__main__":
    unittest.main()