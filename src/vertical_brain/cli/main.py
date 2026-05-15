from __future__ import annotations

import argparse
from pathlib import Path

from vertical_brain.core.context_lock import ContextLock
from vertical_brain.core.models import Chunk, Link
from vertical_brain.core.optimizer import SimpleOptimizer
from vertical_brain.core.router import LLMRouter, StorageModel
from vertical_brain.llm.mock_llm import MockLLM
from vertical_brain.storage.json_store import JsonStore


DEFAULT_MODEL_FILE = Path(__file__).resolve().parents[3] / "data" / "namespaces" / "model.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vb", description="Vertical Brain MVP CLI")
    parser.add_argument("--data-dir", default="data", help="Directory for local JSON storage")
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

    sub.add_parser("tree")

    optimize = sub.add_parser("optimize")
    optimize.add_argument("path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    store = JsonStore(args.data_dir)
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

        chunk = Chunk(
            node_path=decision.target_path,
            content=args.text,
            layer=decision.layer,
            content_type=decision.content_type,
            confidence=decision.confidence,
        )
        store.save_chunk(chunk)

        for peer in decision.peer_links:
            store.save_link(
                Link(
                    source_path=decision.target_path,
                    target_path=peer.path,
                    link_type="peer",
                    reason=peer.reason,
                )
            )

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
        context = lock.build_context(
            decision.target_path,
            include_ancestors=decision.allowed_context.include_ancestors,
            include_peer_links=decision.allowed_context.include_peer_links,
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
        if not context:
            print("(no context found)")
        else:
            for item in context:
                print(f"- {item}")

        print("")
        print("Answer:")
        print(MockLLM().answer_from_context(args.question, context))

    elif args.command == "tree":
        print(store.tree_text())

    elif args.command == "optimize":
        optimizer = SimpleOptimizer(
            store,
            min_compaction_path_parts=storage_model.min_compaction_path_parts,
        )
        print(optimizer.optimize_branch(args.path))


if __name__ == "__main__":
    main()
