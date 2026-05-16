from __future__ import annotations

import argparse
import json
from pathlib import Path

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.context_session import ContextSession
from vertical_brain.core.embedding_router import EmbeddingRouter
from vertical_brain.core.embedding_search import EmbeddingSearch
from vertical_brain.core.json_schema import format_json_schema_errors, validate_json_schema
from vertical_brain.core.models import ContextPolicy, OperationBatchResult
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.core.operations import (
    StorageOperationExecutor,
    operation_batch_from_dict,
    operation_from_route_decision,
)
from vertical_brain.core.router import LLMRouter, StorageModel
from vertical_brain.core.search import BrainSearch
from vertical_brain.llm.embedding import HttpEmbeddingProvider, MockEmbeddingProvider
from vertical_brain.llm.mock_llm import MockLLM
from vertical_brain.storage.json_store import JsonStore
from vertical_brain.storage.sqlite_store import SQLiteStore


DEFAULT_MODEL_FILE = Path(__file__).resolve().parents[3] / "data" / "namespaces" / "model.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vb", description="Vertical Brain MVP CLI")
    parser.add_argument("--data-dir", default="data", help="Directory for local storage")
    parser.add_argument(
        "--storage-backend",
        choices=["json", "sqlite"],
        default="json",
        help="Storage backend to use",
    )
    parser.add_argument(
        "--model-file",
        default=str(DEFAULT_MODEL_FILE),
        help="Storage model JSON used for contracts and format metadata",
    )
    parser.add_argument(
        "--llm-response-file",
        default=None,
        help="Read strict JSON route response from a file instead of a provider",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--route-json", action="store_true", help="Print the route decision as strict JSON")
    ingest.add_argument("text")

    ask = sub.add_parser("ask")
    ask.add_argument("--route-json", action="store_true", help="Print the query route decision as strict JSON")
    ask.add_argument("question")

    namespace_map = sub.add_parser("map")
    namespace_map.add_argument("--path", default=None, help="Limit map to a namespace branch")
    namespace_map.add_argument("--max-depth", type=int, default=None, help="Maximum depth relative to --path")
    namespace_map.add_argument(
        "--summary-chars",
        type=int,
        default=240,
        help="Maximum Gold summary characters per node",
    )
    namespace_map.add_argument("--json", action="store_true", help="Print strict JSON map")

    search = sub.add_parser("search")
    search.add_argument("--path", default=None, help="Limit search to a namespace branch")
    search.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    search.add_argument("--include-stale", action="store_true", help="Include stale and superseded chunks")
    search.add_argument("--semantic", action="store_true", help="Use embedding-based semantic search")
    search.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity score (semantic mode)")
    search.add_argument("--embedding-url", default=None, help="OpenAI-compatible embeddings endpoint URL")
    search.add_argument("--embedding-model", default="nomic-embed-text", help="Embedding model name")
    search.add_argument("--embedding-api-key", default="", help="API key for the embedding endpoint")
    search.add_argument("query")

    context = sub.add_parser("context")
    context_sub = context.add_subparsers(dest="context_command", required=True)
    context_search = context_sub.add_parser("search")
    context_search.add_argument("--path", default=None, help="Limit search to a namespace branch")
    context_search.add_argument("--search-limit", type=int, default=10, help="Maximum candidate handles")
    context_search.add_argument("--context-limit", type=int, default=3, help="Maximum locked contexts to open")
    context_search.add_argument("--items-per-context", type=int, default=6, help="Maximum items per locked context")
    context_search.add_argument("--no-ancestors", action="store_true", help="Do not include ancestor Gold summaries")
    context_search.add_argument(
        "--link-expansion",
        choices=["handles_only", "expanded", "none"],
        default="handles_only",
        help="How to expose horizontal links in locked contexts",
    )
    context_search.add_argument("--json", action="store_true", help="Print strict JSON result")
    context_search.add_argument("query")

    sub.add_parser("tree")

    session_start = sub.add_parser("session-start")
    session_start.add_argument("--path", default=None, help="Limit to a namespace branch")
    session_start.add_argument("--max-depth", type=int, default=None, help="Maximum depth relative to --path")
    session_start.add_argument(
        "--summary-chars",
        type=int,
        default=200,
        help="Maximum Gold summary characters per node",
    )

    route = sub.add_parser("route")
    route.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity score")
    route.add_argument("--limit", type=int, default=5, help="Maximum number of candidates")
    route.add_argument(
        "--embedding-url",
        default=None,
        help="OpenAI-compatible embeddings endpoint URL (e.g. http://localhost:11434/v1/embeddings)",
    )
    route.add_argument("--embedding-model", default="nomic-embed-text", help="Embedding model name")
    route.add_argument("--embedding-api-key", default="", help="API key for the embedding endpoint")
    route.add_argument("text")

    optimize = sub.add_parser("optimize")
    optimize.add_argument("path")

    operation = sub.add_parser("operation")
    operation_sub = operation.add_subparsers(dest="operation_command", required=True)
    operation_dry_run = operation_sub.add_parser("dry-run")
    operation_dry_run.add_argument("file", help="JSON file containing a StorageOperation or StorageOperationBatch")
    operation_apply = operation_sub.add_parser("apply")
    operation_apply.add_argument("file", help="JSON file containing a StorageOperation or StorageOperationBatch")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    store = (
        SQLiteStore(args.data_dir)
        if args.storage_backend == "sqlite"
        else JsonStore(args.data_dir)
    )
    storage_model = StorageModel.load(args.model_file)
    llm = MockLLM.from_response_file(args.llm_response_file) if args.llm_response_file else MockLLM()
    router = LLMRouter(llm, storage_model)
    known_namespaces = [node.path for node in store.list_nodes()]

    if args.command == "ingest":
        decision = router.route_ingest(args.text, known_namespaces=known_namespaces)
        print(f"Target path: {decision.target_path}")
        print(f"Action: {decision.action}")
        print(f"Layer: {decision.layer}")
        print(f"Confidence: {decision.confidence}")
        if args.route_json:
            print("Route decision JSON:")
            print(decision.to_json())

        if router.requires_clarification(decision):
            print("Clarification needed: provide a more specific domain or namespace hint.")
            return

        operation = operation_from_route_decision(decision, args.text)
        StorageOperationExecutor(store).apply(operation)

        if decision.peer_links:
            print("Peer links:")
            for peer in decision.peer_links:
                print(f"- {peer.path}: {peer.reason}")
        if decision.stale_candidates:
            print("Stale candidates:")
            for candidate in decision.stale_candidates:
                print(f"- {candidate.path}: {candidate.reason}")

    elif args.command == "ask":
        decision = router.route_query(args.question, known_namespaces=known_namespaces)
        lock = ContextLock(store)
        locked_context = lock.open_locked_context(
            decision.target_path,
            policy=ContextPolicy(
                include_ancestors=decision.allowed_context.include_ancestors,
                include_target=True,
                link_expansion="handles_only" if decision.allowed_context.include_peer_links else "none",
            ),
        )

        print(f"Target path: {decision.target_path}")
        print(f"Query type: {decision.query_type}")
        print(f"Confidence: {decision.confidence}")
        if args.route_json:
            print("Route decision JSON:")
            print(decision.to_json())
        if router.requires_clarification(decision):
            print("Clarification needed: provide a more specific domain or namespace hint.")
            return
        print(
            "Allowed context policy: "
            f"ancestors={decision.allowed_context.include_ancestors}, "
            f"peer_links={decision.allowed_context.include_peer_links}, "
            f"exclude_other_branches={decision.allowed_context.exclude_other_branches}"
        )
        print("Allowed context:")
        if not locked_context.items:
            print("(no context found)")
        else:
            for item in locked_context.items:
                print(f"- [{item.path}][{item.layer}] {item.content}")
        if locked_context.link_handles:
            print("Available link handles:")
            for handle in locked_context.link_handles:
                print(f"- {handle.link_id} -> {handle.target_path} ({handle.link_type}): {handle.reason}")
        if locked_context.omitted_items:
            print(f"Omitted items due to budget: {locked_context.omitted_items}")

        print("")
        print("Answer:")
        print(MockLLM().answer_from_context(args.question, locked_context.as_prompt_lines()))

    elif args.command == "session-start":
        print(ContextSession(store).session_prompt(
            root_path=args.path,
            max_depth=args.max_depth,
            summary_max_chars=args.summary_chars,
        ))

    elif args.command == "tree":
        print(store.tree_text())

    elif args.command == "map":
        namespace_map = ContextSession(store).namespace_map(
            root_path=args.path,
            max_depth=args.max_depth,
            summary_max_chars=args.summary_chars,
        )
        if args.json:
            print(namespace_map.to_json())
        else:
            print("Namespace map:")
            if not namespace_map.nodes:
                print("(empty map)")
            else:
                for node in namespace_map.nodes:
                    indent = "  " * node.depth
                    counts = (
                        f"chunks={node.chunk_count}, active={node.active_chunk_count}, "
                        f"subtree={node.subtree_chunk_count}, links={node.link_count}"
                    )
                    print(f"{indent}- {node.path} ({counts})")
                    if node.gold_summary:
                        print(f"{indent}  gold: {node.gold_summary}")
                    if node.children:
                        print(f"{indent}  children: {', '.join(node.children)}")
                    if node.link_handles:
                        for handle in node.link_handles:
                            print(
                                f"{indent}  link: {handle.link_id} -> {handle.target_path} "
                                f"({handle.link_type})"
                            )
            if namespace_map.omitted_nodes:
                print(f"Omitted nodes due to depth limit: {namespace_map.omitted_nodes}")

    elif args.command == "search":
        if args.semantic:
            provider = (
                HttpEmbeddingProvider(
                    url=args.embedding_url,
                    model=args.embedding_model,
                    api_key=args.embedding_api_key,
                )
                if args.embedding_url
                else MockEmbeddingProvider()
            )
            results = EmbeddingSearch(store, provider).search(
                args.query,
                root_path=args.path,
                limit=args.limit,
                include_stale=args.include_stale,
                threshold=args.threshold,
            )
        else:
            results = BrainSearch(store).search(
                args.query,
                root_path=args.path,
                limit=args.limit,
                include_stale=args.include_stale,
            )
        print("Search results:")
        if not results:
            print("(no results)")
        else:
            for result in results:
                metadata = result.source
                if result.layer and result.content_type:
                    metadata = f"{metadata}/{result.layer}/{result.content_type}"
                print(f"- score={result.score:.6f} [{result.path}][{metadata}] {result.snippet}")

    elif args.command == "context":
        if args.context_command == "search":
            result = ContextSession(store).search_locked_context(
                args.query,
                root_path=args.path,
                search_limit=args.search_limit,
                context_limit=args.context_limit,
                items_per_context=args.items_per_context,
                include_ancestors=not args.no_ancestors,
                link_expansion=args.link_expansion,
            )
            if args.json:
                print(result.to_json())
            else:
                print("Candidate handles:")
                if not result.candidate_handles:
                    print("(no candidates)")
                else:
                    for handle in result.candidate_handles:
                        metadata = handle.source
                        if handle.layer and handle.content_type:
                            metadata = f"{metadata}/{handle.layer}/{handle.content_type}"
                        print(f"- score={handle.score:.6f} [{handle.path}][{metadata}]")

                print("Locked contexts:")
                if not result.locked_contexts:
                    print("(no locked context)")
                else:
                    for locked_context in result.locked_contexts:
                        print(f"Context: {locked_context.target_path}")
                        if not locked_context.items:
                            print("- (no context items)")
                        else:
                            for item in locked_context.items:
                                print(f"- [{item.path}][{item.layer}] {item.content}")
                        if locked_context.link_handles:
                            print("Available link handles:")
                            for handle in locked_context.link_handles:
                                print(
                                    f"- {handle.link_id} -> {handle.target_path} "
                                    f"({handle.link_type}): {handle.reason}"
                                )
                        if locked_context.omitted_items:
                            print(f"Omitted items due to budget: {locked_context.omitted_items}")
                if result.omitted_candidates:
                    print(f"Omitted candidate paths: {result.omitted_candidates}")

    elif args.command == "route":
        provider = (
            HttpEmbeddingProvider(
                url=args.embedding_url,
                model=args.embedding_model,
                api_key=args.embedding_api_key,
            )
            if args.embedding_url
            else MockEmbeddingProvider()
        )
        candidates = EmbeddingRouter(store, provider).find_candidates(
            args.text,
            threshold=args.threshold,
            limit=args.limit,
        )
        if not candidates:
            print("(no matching namespaces)")
        else:
            for c in candidates:
                print(f"score={c.score:.4f}  {c.path}  —  {c.gold_summary}")

    elif args.command == "optimize":
        optimizer = SimpleOptimizer(
            store,
            min_compaction_path_parts=storage_model.min_compaction_path_parts,
        )
        print(optimizer.optimize_branch(args.path))

    elif args.command == "operation":
        payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Operation file must contain a JSON object")
        schema_validation = validate_json_schema(payload, storage_model.storage_operation_payload_schema)
        if not schema_validation.valid:
            if args.operation_command == "dry-run":
                print(OperationBatchResult(status="invalid", validation=schema_validation).to_json())
                return
            raise ValueError(format_json_schema_errors(schema_validation))
        batch = operation_batch_from_dict(payload)
        executor = StorageOperationExecutor(store)
        result = executor.dry_run_batch(batch) if args.operation_command == "dry-run" else executor.apply_batch(batch)
        print(result.to_json())


if __name__ == "__main__":
    main()
