from vertical_brain.core.models import Chunk, StorageOperationBatch
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.storage.json_store import JsonStore


def _seeded_store(tmp_path):
    store = JsonStore(tmp_path)
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="fact one"))
    store.save_chunk(Chunk(node_path="WORK/DataArt/Databricks", content="fact two"))
    return store


def test_plan_returns_operation_batch(tmp_path):
    store = _seeded_store(tmp_path)
    plan = SimpleOptimizer(store, min_compaction_path_parts=3).plan_branch("WORK")
    assert isinstance(plan, StorageOperationBatch)
    assert plan.operations


def test_plan_does_not_mutate_store(tmp_path):
    store = _seeded_store(tmp_path)
    before = [(c.id, c.status, c.content) for c in store.list_chunks()]
    SimpleOptimizer(store, min_compaction_path_parts=3).plan_branch("WORK")
    after = [(c.id, c.status, c.content) for c in store.list_chunks()]
    assert before == after


def test_plan_matches_what_optimize_would_apply(tmp_path):
    store = _seeded_store(tmp_path)
    plan = SimpleOptimizer(store, min_compaction_path_parts=3).plan_branch("WORK")
    op_types = [op.operation for op in plan.operations]
    assert "append_chunk" in op_types
    assert "supersede_chunk" in op_types


# ── optimize_branch version metadata ─────────────────────────────────────────

def test_optimize_branch_applies_batch_with_branch_version(tmp_path):
    """optimize_branch must stamp branch_path/start_version so OCC is enforced."""
    from unittest.mock import patch
    store = _seeded_store(tmp_path)
    store.ensure_node("WORK")
    opt = SimpleOptimizer(store, min_compaction_path_parts=3)

    # Capture version snapshot BEFORE optimize_branch runs.
    version_before = store.get_node("WORK").version
    captured = []

    original_apply_batch = __import__(
        "vertical_brain.core.operations", fromlist=["StorageOperationExecutor"]
    ).StorageOperationExecutor.apply_batch

    def _spy(self, batch):
        captured.append(batch)
        return original_apply_batch(self, batch)

    with patch(
        "vertical_brain.core.optimizer.StorageOperationExecutor.apply_batch",
        new=_spy,
    ):
        opt.optimize_branch("WORK")

    assert len(captured) == 1
    assert captured[0].branch_path == "WORK"
    assert captured[0].start_version == version_before


def test_decay_disabled_list_links_not_called(tmp_path):
    """With decay_rate >= 1.0, list_links must not be called."""
    from unittest.mock import patch
    store = _seeded_store(tmp_path)
    opt = SimpleOptimizer(store, min_compaction_path_parts=3, decay_rate=1.0)

    with patch.object(store, "list_links", wraps=store.list_links) as mock_links:
        opt.plan_branch("WORK")
        mock_links.assert_not_called()


def test_decay_enabled_list_links_called_exactly_once(tmp_path):
    """With decay_rate < 1.0, list_links must be called exactly once per snapshot."""
    from unittest.mock import patch
    store = _seeded_store(tmp_path)
    opt = SimpleOptimizer(store, min_compaction_path_parts=3, decay_rate=0.5, decay_days=1)

    with patch.object(store, "list_links", wraps=store.list_links) as mock_links:
        opt.plan_branch("WORK")
        assert mock_links.call_count == 1


def test_get_chunks_by_path_called_exactly_once(tmp_path):
    """plan_branch must issue exactly one get_chunks_by_path call."""
    from unittest.mock import patch
    store = _seeded_store(tmp_path)
    opt = SimpleOptimizer(store, min_compaction_path_parts=3)

    with patch.object(store, "get_chunks_by_path", wraps=store.get_chunks_by_path) as mock_chunks:
        opt.plan_branch("WORK")
        assert mock_chunks.call_count == 1
