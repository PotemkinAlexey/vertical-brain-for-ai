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
