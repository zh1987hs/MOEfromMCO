from mnox_retrieval.simulate import simulate_dataset


def test_simulation_counts(tmp_path):
    res = simulate_dataset(tmp_path, n_positives=20, n_clusters=4, unlabeled_size=200, easy_hits=30, hidden_positives=10)
    assert len(res.positives) == 20
    assert len(res.unlabeled) == 200
    assert int(res.unlabeled["is_hidden_positive"].sum()) == 10
