from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "stability_remote_candidates.py"
    spec = importlib.util.spec_from_file_location("stability_remote_candidates", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stability = _load_module()


class StabilityRemoteCandidatesSmokeTest(unittest.TestCase):
    def _toy_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "candidate_id": ["c1", "c2", "c3"],
                "nearest_positive_family": ["mcoA", "mcoA", "other"],
                "best_identity_to_positive": [0.18, 0.24, 0.21],
                "embedding_similarity": [0.82, 0.71, 0.66],
                "positive_support_score": [0.77, 0.61, 0.55],
                "local_density_score": [0.33, 0.40, 0.42],
                "novelty_score": [0.24, 0.35, 0.30],
                "false_positive_risk": [0.12, 0.21, 0.18],
                "hmm_best_evalue": [1e-8, 1e-5, 1e-6],
                "hmm_best_bitscore": [42.0, 28.0, 33.0],
                "remote_rank": [1, 2, 3],
            }
        )

    def test_analyze_stability_outputs_expected_columns(self) -> None:
        summary = stability.analyze_stability(
            df=self._toy_df(),
            target_family="mcoA",
            n_iter=20,
            seed=42,
            topk_list=[10, 20, 50],
            base_weights=stability.DEFAULT_BASE_WEIGHTS,
            perturb_range=stability.DEFAULT_PERTURB_RANGE,
        )
        expected = {
            "candidate_id",
            "nearest_positive_family",
            "best_identity_to_positive",
            "embedding_similarity",
            "positive_support_score",
            "local_density_score",
            "novelty_score",
            "false_positive_risk",
            "baseline_remote_rank",
            "baseline_rank_recomputed",
            "top10_freq",
            "top20_freq",
            "top50_freq",
            "mean_rank",
            "std_rank",
            "best_rank",
            "worst_rank",
        }
        self.assertTrue(expected.issubset(set(summary.columns)))
        self.assertEqual(summary["nearest_positive_family"].nunique(), 1)
        self.assertEqual(summary["nearest_positive_family"].iloc[0], "mcoA")

    def test_empty_family_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No remote candidates found"):
            stability.analyze_stability(
                df=self._toy_df(),
                target_family="missing_family",
                n_iter=10,
                seed=42,
                topk_list=[10, 20],
                base_weights=stability.DEFAULT_BASE_WEIGHTS,
                perturb_range=stability.DEFAULT_PERTURB_RANGE,
            )

    def test_cli_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            input_csv = tmpdir / "ranked_candidates_remote_only.csv"
            out_csv = tmpdir / "stability.csv"
            out_json = tmpdir / "stability.json"
            self._toy_df().to_csv(input_csv, index=False)

            rc = stability.main(
                [
                    "--run-dir",
                    str(tmpdir),
                    "--target-family",
                    "mcoA",
                    "--n-iter",
                    "10",
                    "--seed",
                    "42",
                    "--out-csv",
                    str(out_csv),
                    "--out-json",
                    str(out_json),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(out_csv.exists())
            self.assertTrue(out_json.exists())
            meta = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(meta["target_family"], "mcoA")


if __name__ == "__main__":
    unittest.main()
