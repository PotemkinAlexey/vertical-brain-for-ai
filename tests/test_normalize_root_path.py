import pytest

from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.models import Chunk
from vertical_brain.core.namespace_map import normalize_namespace_root_path
from vertical_brain.storage.json_store import JsonStore


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("/", None),
        ("WORK", "WORK"),
        ("  WORK/DataArt  ", "WORK/DataArt"),
    ],
)
def test_normalize_namespace_root_path(raw: str | None, expected: str | None) -> None:
    assert normalize_namespace_root_path(raw) == expected


def test_session_prompt_slash_root_path_shows_all_namespaces(tmp_path) -> None:
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt", content="fact"))
    store.save_chunk(Chunk(node_path="PERSONAL/Blog", content="post"))

    prompt = ContextSession(store).session_prompt(root_path="/")

    assert "WORK/DataArt" in prompt
    assert "PERSONAL/Blog" in prompt
    assert "0 nodes · 0 active chunks" not in prompt
