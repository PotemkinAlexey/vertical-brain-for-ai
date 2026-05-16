"""ThreadLocalSQLiteStoreProxy tests."""
from __future__ import annotations

import threading

from vertical_brain.core.models import Chunk
from vertical_brain.storage.thread_local_store import ThreadLocalSQLiteStoreProxy


def test_proxy_forwards_basic_operations(tmp_path):
    proxy = ThreadLocalSQLiteStoreProxy(tmp_path)
    proxy.ensure_node("WORK/Project")
    proxy.save_chunk(Chunk(node_path="WORK/Project", content="fact"))

    chunks = proxy.get_chunks_by_path("WORK/Project")
    assert len(chunks) == 1
    assert chunks[0].content == "fact"


def test_proxy_each_thread_gets_own_store_instance(tmp_path):
    proxy = ThreadLocalSQLiteStoreProxy(tmp_path)
    instances: list[object] = []
    lock = threading.Lock()

    def _capture():
        store = proxy._store()
        with lock:
            instances.append(id(store))

    threads = [threading.Thread(target=_capture) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each thread must have its own SQLiteStore instance.
    assert len(set(instances)) == 3


def test_proxy_shares_same_database_across_threads(tmp_path):
    """Writes from one thread must be visible to reads from another."""
    proxy = ThreadLocalSQLiteStoreProxy(tmp_path)
    proxy.ensure_node("WORK/A")
    proxy.save_chunk(Chunk(node_path="WORK/A", content="shared fact"))

    results: list[list] = []

    def _read():
        chunks = proxy.get_chunks_by_path("WORK/A")
        results.append(chunks)

    t = threading.Thread(target=_read)
    t.start()
    t.join()

    assert len(results[0]) == 1
    assert results[0][0].content == "shared fact"


def test_proxy_repr_is_informative(tmp_path):
    proxy = ThreadLocalSQLiteStoreProxy(tmp_path, "brain.sqlite")
    assert "ThreadLocalSQLiteStoreProxy" in repr(proxy)
    assert "brain.sqlite" in repr(proxy)


def test_proxy_same_thread_gets_same_instance(tmp_path):
    proxy = ThreadLocalSQLiteStoreProxy(tmp_path)
    assert proxy._store() is proxy._store()
