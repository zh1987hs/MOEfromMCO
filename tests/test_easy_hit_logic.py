from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mnox.retrieval import build_remote_candidate_pool, decide_easy_hits, rank_remote_candidates
from mnox.io_fasta import read_fasta_records
from run_pipeline import _build_remote_experimental_view, _export_top_ranked_subset


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


class RemoteDiscoveryLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.remote_cfg = {
            "enabled": True,
            "identity_max": 0.30,
            "qcov_min": 0.60,
            "tcov_min": 0.60,
            "require_missed_only": True,
            "hmm_support": {
                "enabled": True,
                "evalue_max": 1e-3,
                "bitscore_min": 0.0,
                "use_as_soft_support": True,
            },
            "ranking": {
                "mode": "remote_score",
                "w_affinity": 0.35,
                "w_positive_support": 0.20,
                "w_local_density": 0.15,
                "w_novelty": 0.30,
                "w_false_positive_risk": -0.15,
                "w_identity_penalty": -0.20,
                "w_hmm_support": 0.05,
            },
        }

    def test_remote_pool_enforces_identity_and_coverage_gates(self) -> None:
        ranked = pd.DataFrame(
            {
                "candidate_id": ["hi_identity", "remote_good", "remote_good_2", "low_qcov", "easy_remote"],
                "easy_or_missed": ["missed", "missed", "missed", "missed", "easy"],
                "mmseqs_best_fident": [0.35, 0.18, 0.22, 0.12, 0.18],
                "mmseqs_best_qcov": [0.90, 0.75, 0.82, 0.50, 0.88],
                "mmseqs_best_tcov": [0.90, 0.70, 0.76, 0.80, 0.90],
                "embedding_similarity": [0.20, 0.80, 0.65, 0.70, 0.90],
                "positive_support_score": [0.20, 0.70, 0.62, 0.60, 0.80],
                "local_density_score": [0.30, 0.40, 0.45, 0.30, 0.30],
                "novelty_score": [0.95, 0.65, 0.72, 0.90, 0.70],
                "false_positive_risk": [0.10, 0.20, 0.18, 0.15, 0.05],
                "hmm_best_evalue": [1e-2, 1e-6, 1e-4, 1e-5, 1e-8],
                "hmm_best_bitscore": [5.0, 30.0, 25.0, 20.0, 40.0],
                "confidence_tier": ["medium", "high", "medium", "medium", "high"],
                "flags": ["", "", "", "", ""],
                "reason_for_high_rank": ["novel", "supported_remote", "supported_remote", "low_cov", "easy_hit"],
                "nearest_positive_family": ["F1", "F1", "F2", "F2", "F3"],
            }
        )

        remote_df, diag, _ = build_remote_candidate_pool(ranked, self.remote_cfg)
        remote_ranked = rank_remote_candidates(remote_df, self.remote_cfg)

        self.assertEqual(remote_df["candidate_id"].tolist(), ["remote_good", "remote_good_2"])
        self.assertTrue((remote_df["best_identity_to_positive"] < 0.30).all())
        self.assertTrue((remote_df["mmseqs_best_qcov"] >= 0.60).all())
        self.assertTrue((remote_df["mmseqs_best_tcov"] >= 0.60).all())
        self.assertEqual(int(diag["filtered_by_identity"]), 1)
        self.assertEqual(int(diag["filtered_by_qcov"]), 1)
        self.assertEqual(int(diag["filtered_by_easy"]), 1)
        self.assertTrue(remote_ranked["remote_rank"].tolist() == [1, 2])
        self.assertEqual(remote_ranked["remote_rank"].nunique(), len(remote_ranked))
        self.assertTrue((remote_ranked["easy_or_missed"] == "missed").all())

    def test_high_novelty_but_identity_35_cannot_enter_remote(self) -> None:
        ranked = pd.DataFrame(
            {
                "candidate_id": ["novel_but_too_close"],
                "easy_or_missed": ["missed"],
                "mmseqs_best_fident": [0.35],
                "mmseqs_best_qcov": [0.95],
                "mmseqs_best_tcov": [0.95],
                "embedding_similarity": [0.50],
                "positive_support_score": [0.50],
                "local_density_score": [0.30],
                "novelty_score": [0.99],
                "false_positive_risk": [0.05],
                "hmm_best_evalue": [1e-8],
                "hmm_best_bitscore": [50.0],
            }
        )
        remote_df, _, _ = build_remote_candidate_pool(ranked, self.remote_cfg)
        self.assertTrue(remote_df.empty)

    def test_remote_experimental_view_tracks_remote_rank_order(self) -> None:
        remote_ranked = pd.DataFrame(
            {
                "remote_rank": [2, 1],
                "candidate_id": ["r2", "r1"],
                "nearest_positive_family": ["F2", "F1"],
                "best_identity_to_positive": [0.22, 0.18],
                "mmseqs_best_qcov": [0.75, 0.88],
                "mmseqs_best_tcov": [0.74, 0.80],
                "embedding_similarity": [0.70, 0.81],
                "positive_support_score": [0.62, 0.77],
                "local_density_score": [0.45, 0.35],
                "novelty_score": [0.73, 0.69],
                "hmm_best_evalue": [1e-4, 1e-7],
                "hmm_best_bitscore": [26.0, 40.0],
                "false_positive_risk": [0.18, 0.12],
                "confidence_tier": ["medium", "high"],
                "flags": ["flag_b", "flag_a"],
                "reason_for_high_rank": ["remote_b", "remote_a"],
                "remote_score": [0.71, 0.84],
            }
        )

        exp_view = _build_remote_experimental_view(remote_ranked)
        remote_exp_view = _build_remote_experimental_view(remote_ranked)
        self.assertEqual(exp_view["candidate_id"].tolist(), ["r1", "r2"])
        self.assertEqual(exp_view["remote_rank"].tolist(), [1, 2])
        self.assertEqual(exp_view["experimental_priority_rank"].tolist(), [1, 2])
        self.assertEqual(exp_view["candidate_id"].head(2).tolist(), remote_exp_view["candidate_id"].head(2).tolist())
        self.assertIn("risk_flags", exp_view.columns)
        self.assertNotIn("flags", exp_view.columns)

    def test_top_remote_fasta_count_matches_top_n_or_remote_size(self) -> None:
        ranked_remote = pd.DataFrame(
            {
                "candidate_id": ["r1", "r2", "r3"],
                "remote_rank": [1, 2, 3],
                "remote_score": [0.9, 0.8, 0.7],
            }
        )
        seqs = {"r1": "MAAA", "r2": "MBBB", "r3": "MCCC"}

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            _export_top_ranked_subset(
                ranked_remote,
                seqs,
                tmpdir / "top_remote_candidates.csv",
                tmpdir / "top_remote_candidates.fasta",
                top_n=2,
                make_fasta=True,
                rank_col="remote_rank",
            )
            recs = read_fasta_records(tmpdir / "top_remote_candidates.fasta")
            self.assertEqual(len(recs), 2)


if __name__ == "__main__":
    unittest.main()
