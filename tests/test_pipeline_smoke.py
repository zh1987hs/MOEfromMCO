from pathlib import Path

from mnox_retrieval.config import Config
from mnox_retrieval.pipeline import load_or_simulate, run_once


def test_pipeline_smoke(tmp_path):
    cfg = Config(
        {
            "seed": 1,
            "simulation": {
                "n_positives": 20,
                "n_clusters": 4,
                "unlabeled_size": 150,
                "easy_hits": 20,
                "hidden_positives": 10,
                "length_min": 90,
                "length_max": 140,
            },
            "embedding": {"mode": "fallback", "dim": 16},
            "clustering": {"mode": "embedding", "n_clusters": 4},
            "scoring": {
                "weights": {"positive_affinity": 0.45, "novelty": 0.25, "local_support": 0.20, "easy_hit_penalty": 0.10},
                "easy_hit_threshold": 0.7,
                "local_support_neighbors": 8,
            },
            "evaluation": {"topk": [10, 20]},
        }
    )
    sim_dir = tmp_path / "sim"
    out_dir = tmp_path / "out"
    pos, unl = load_or_simulate(sim_dir, cfg)
    ranked, metrics = run_once(pos, unl, cfg, out_dir)
    assert len(ranked) == len(unl)
    assert "Recall@10" in metrics
    assert (out_dir / "ranked_candidates.csv").exists()
