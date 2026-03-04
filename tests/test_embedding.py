import pandas as pd

from mnox_retrieval.embedding import compute_embeddings


def test_compute_embeddings_accepts_pooling_kwarg():
    df = pd.DataFrame({"id": ["x", "y"], "sequence": ["ACDEFGHIK", "LMNPQRSTV"]})
    emb = compute_embeddings(df, mode="fallback", dim=4, pooling="mean")
    assert emb.shape[0] == 2
    assert emb.shape[1] >= 2
