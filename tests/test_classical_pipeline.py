import unittest

import pandas as pd

from src.leak_audit import connected_groups, dataset_hash
from src.features import NUM_COLS
from src.train import MODEL_NUM_COLS


class GroupingTest(unittest.TestCase):
    def test_model_uses_twelve_known_engineered_features(self):
        expected = [
            "urgency_kw_count", "scarcity_kw_count", "shame_phrase_flag",
            "cancel_diff_score", "social_proof_flag", "price_drip_flag",
            "discount_claim_flag", "neg_option_flag", "exclamation_count",
            "question_count", "number_present", "time_reference_flag",
        ]
        self.assertEqual(NUM_COLS, expected)
        self.assertEqual(MODEL_NUM_COLS, expected)

    def test_page_and_skeleton_links_are_transitive(self):
        rows = pd.DataFrame({
            "page_id": ["page-a", "page-a", "page-b", "page-c"],
            "text": ["ordinary text", "Only 2 left", "Only 3 left", "separate text"],
            "label": [0, 1, 1, 0],
            "Pattern Category": ["safe", "urgency", "urgency", "safe"],
        })
        groups = connected_groups(rows)

        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])
        self.assertNotEqual(dataset_hash(rows), dataset_hash(rows.iloc[::-1]))


if __name__ == "__main__":
    unittest.main()
