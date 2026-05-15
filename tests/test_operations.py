from vertical_brain.core.models import (
    ChunkInput,
    LinkInput,
    StaleCandidateInput,
    StorageOperation,
)
from vertical_brain.core.operations import StorageOperationExecutor
from vertical_brain.storage.json_store import JsonStore


def test_append_chunk_operation_writes_chunk_links_and_stale_candidates(tmp_path):
    store = JsonStore(tmp_path)
    operation = StorageOperation(
        operation="append_chunk",
        target_path="WORK/Vertical/Node",
        chunk=ChunkInput(
            content="A durable fact.",
            layer="silver",
            content_type="fact",
            confidence=0.91,
        ),
        links=[
            LinkInput(
                target_path="WORK/Other/Node",
                link_type="peer",
                reason="Explicit horizontal link.",
            )
        ],
        stale_candidates=[
            StaleCandidateInput(
                path="WORK/Vertical/Node",
                reason="Model identified a possible stale chunk.",
            )
        ],
        confidence=0.91,
        reasoning_summary="Model requested append_chunk.",
    )

    result = StorageOperationExecutor(store).apply(operation)

    chunks = store.get_chunks_by_path("WORK/Vertical/Node")
    assert result.operation == "append_chunk"
    assert result.chunk_id == chunks[0].id
    assert chunks[0].content == "A durable fact."
    assert chunks[0].source == "model"
    assert store.get_peer_paths("WORK/Vertical/Node") == ["WORK/Other/Node"]
    assert result.stale_candidates[0].path == "WORK/Vertical/Node"


def test_create_node_operation_creates_ancestor_chain(tmp_path):
    store = JsonStore(tmp_path)
    operation = StorageOperation(operation="create_node", target_path="WORK/New/Branch")

    result = StorageOperationExecutor(store).apply(operation)

    assert result.status == "applied"
    assert store.get_node("WORK/New/Branch") is not None
