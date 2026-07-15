import unittest

from hf_space.presentation import result_status


class PresentationTest(unittest.TestCase):
    def test_low_scores_abstain_instead_of_becoming_benign(self):
        self.assertEqual(result_status("False Urgency", 0.49), "inconclusive")
        self.assertEqual(result_status("Not a Dark Pattern", 0.49), "inconclusive")
        self.assertEqual(result_status("False Urgency", 0.50), "signal")
        self.assertEqual(result_status("Not a Dark Pattern", 0.50), "no_signal")


if __name__ == "__main__":
    unittest.main()
