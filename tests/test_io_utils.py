from mnox_retrieval.io_utils import read_fasta, write_fasta


def test_write_then_read_fasta(tmp_path):
    out = tmp_path / "x.fasta"
    write_fasta([("a", "ACDEFG"), ("b", "MNPQRS")], out)
    rows = list(read_fasta(out))
    assert len(rows) == 2
    assert rows[0][0] == "a"
    assert rows[0][1] == "ACDEFG"
