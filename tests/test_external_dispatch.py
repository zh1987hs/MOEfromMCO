from mnox_retrieval.external_tools import detect_tools
from mnox_retrieval.installer import get_install_commands


def test_detect_tools_keys():
    tools = detect_tools()
    assert set(tools.keys()) == {"blastp", "makeblastdb", "phmmer"}


def test_install_commands():
    cmds = get_install_commands("conda")
    assert any("blast" in c for c in cmds)
