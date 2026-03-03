import numpy as np
import pandas as pd

from mnox_retrieval.scoring import combine_scores


def test_combine_scores_orders():
    df = pd.DataFrame(
        {
            "positive_affinity": [0.9, 0.5],
            "novelty_score": [0.8, 0.2],
            "local_support_score": [0.7, 0.6],
            "easy_hit_penalty": [0.0, 1.0],
        }
    )
    ranked = combine_scores(df, 0.5, 0.3, 0.2, 0.2)
    assert ranked.iloc[0]["rank"] == 1
    assert np.isfinite(ranked["final_score"]).all()
