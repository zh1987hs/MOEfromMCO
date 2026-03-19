from __future__ import annotations

import unittest

import pandas as pd

from mnox.retrieval import decide_easy_hits


class EasyHitLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.easy_cfg = {
            "decision_rule": "consensus",
            "profile": "balanced",
            "max_easy_fraction": 0.30,
            "fallback_if_too_many_easy": "tighten",
            "calibrate": False,
            "target_easy_fraction_max": 0.30,
        }

    def test_or_mode_matches_legacy_union(self) -> None:
        mm = pd.DataFrame(
            {
                "candidate_id": ["a", "b"],
                "mmseqs_hit_balanced": [True, False],
                "mmseqs_hit_strict": [True, False],
                "mmseqs_strength_balanced": [0.9, 0.0],
                "mmseqs_strength_strict": [0.9, 0.0],
            }
        )
        hmm = pd.DataFrame(
            {
                "candidate_id": ["a", "b"],
                "hmm_hit_balanced": [False, True],
                "hmm_hit_strict": [False, True],
                "hmm_strength_balanced": [0.0, 0.8],
                "hmm_strength_strict": [0.0, 0.8],
            }
        )
        decided, _ = decide_easy_hits(["a", "b"], mm, hmm, {**self.easy_cfg, "decision_rule": "or"})
        self.assertEqual(decided["easy_hit_flag"].tolist(), [True, True])

    def test_consensus_rejects_single_tool_balanced_edge(self) -> None:
        mm = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "mmseqs_hit_balanced": [True],
                "mmseqs_hit_strict": [False],
                "mmseqs_strength_balanced": [0.7],
                "mmseqs_strength_strict": [0.2],
            }
        )
        hmm = pd.DataFrame(
            {
                "candidate_id": ["a"],
                "hmm_hit_balanced": [False],
                "hmm_hit_strict": [False],
                "hmm_strength_balanced": [0.0],
                "hmm_strength_strict": [0.0],
            }
        )
        decided, _ = decide_easy_hits(["a"], mm, hmm, self.easy_cfg)
        self.assertFalse(bool(decided.loc[0, "easy_hit_flag"]))

    def test_tighten_reduces_easy_fraction_when_hmm_is_too_broad(self) -> None:
        candidate_ids = [f"c{i}" for i in range(10)]
        mm = pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "mmseqs_hit_balanced": [False] * 10,
                "mmseqs_hit_strict": [False] * 10,
                "mmseqs_strength_balanced": [0.0] * 10,
                "mmseqs_strength_strict": [0.0] * 10,
            }
        )
        hmm = pd.DataFrame(
            {
                "candidate_id": candidate_ids,
                "hmm_hit_loose": [True] * 10,
                "hmm_hit_balanced": [True] * 8 + [False, False],
                "hmm_hit_strict": [True] * 2 + [False] * 8,
                "hmm_strength_loose": [0.9] * 10,
                "hmm_strength_balanced": [0.8] * 8 + [0.0, 0.0],
                "hmm_strength_strict": [0.95] * 2 + [0.0] * 8,
            }
        )
        decided, diag = decide_easy_hits(candidate_ids, mm, hmm, {**self.easy_cfg, "calibrate": True})
        self.assertLessEqual(float(diag["easy_fraction"]), 0.30)
        self.assertEqual(int((decided["easy_confidence_class"] == "hmm_strict_only").sum()), 2)


if __name__ == "__main__":
    unittest.main()
