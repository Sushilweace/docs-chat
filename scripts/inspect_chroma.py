"""Inspect the ChromaDB vector store (read-only helper).

Examples (run from the project root, with the venv active):

    python scripts/inspect_chroma.py                       # summary per user/document
    python scripts/inspect_chroma.py --show 3              # + first 3 chunks of every document
    python scripts/inspect_chroma.py --doc 2 --show 10 --full
    python scripts/inspect_chroma.py --query "tech skills" --user 2
                                                           # what would retrieval return?

It never writes. ``--query`` loads the embedding model (a few seconds) and prints each
hit's cosine distance so you can compare it with DISTANCE_THRESHOLD.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `app` importable

import chromadb  # noqa: E402

from app.config import get_settings  # noqa: E402


def _short(text: str, full: bool, width: int = 110) -> str:
    text = " ".join(text.split())
    return text if full or len(text) <= width else text[: width - 1] + "…"


def _where(user: int | None, doc: int | None) -> dict | None:
    clauses = []
    if user is not None:
        clauses.append({"user_id": user})
    if doc is not None:
        clauses.append({"document_id": doc})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", type=int, help="only this user id")
    parser.add_argument("--doc", type=int, help="only this document id")
    parser.add_argument("--show", type=int, default=0, metavar="N", help="print the first N chunks per document")
    parser.add_argument("--full", action="store_true", help="don't truncate chunk text")
    parser.add_argument("--query", help="run a similarity search and show distances")
    parser.add_argument("--top-k", type=int, help="results for --query (default: TOP_K setting)")
    args = parser.parse_args()

    # Windows consoles often default to cp1252; PDFs contain bullets, dashes, etc.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        collection = client.get_collection(settings.chroma_collection)
    except Exception:  # noqa: BLE001 - chroma raises different errors across versions
        sys.exit(f"No collection {settings.chroma_collection!r} in {settings.chroma_path} (nothing ingested yet?)")

    where = _where(args.user, args.doc)
    data = collection.get(where=where, include=["metadatas", "documents"])
    total_all = collection.count()
    print(f"Store: {settings.chroma_path}  |  collection: {settings.chroma_collection}")
    print(f"Chunks: {len(data['ids'])} shown of {total_all} total  |  distance metric: "
          f"{(collection.metadata or {}).get('hnsw:space', 'l2')}")
    sample = collection.get(limit=1, include=["embeddings"])
    if len(sample["ids"]):
        print(f"Embedding dimension: {len(sample['embeddings'][0])}")

    # Group chunks by (user, document).
    groups: dict[tuple, list[tuple[dict, str]]] = defaultdict(list)
    for meta, text in zip(data["metadatas"], data["documents"]):
        groups[(meta.get("user_id"), meta.get("document_id"), meta.get("source"))].append((meta, text))

    print("\nuser  doc  chunks  pages  source")
    print("----  ---  ------  -----  ------")
    for (user_id, doc_id, source), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        pages = sorted({m.get("page") for m, _ in items if m.get("page") is not None})
        page_str = f"{pages[0]}-{pages[-1]}" if pages else "-"
        print(f"{user_id!s:>4}  {doc_id!s:>3}  {len(items):>6}  {page_str:>5}  {source}")

        for meta, text in sorted(items, key=lambda it: (it[0].get("page", 0), it[0].get("chunk_index", 0)))[: args.show]:
            print(f"        p{meta.get('page')}#{meta.get('chunk_index')} ({len(text)} chars): {_short(text, args.full)}")

    if args.query:
        from app.services import embeddings  # imported lazily: loads the model

        top_k = args.top_k or settings.top_k
        vector = embeddings.embed_texts([args.query])[0]
        result = collection.query(
            query_embeddings=[vector], n_results=top_k, where=where, include=["metadatas", "documents", "distances"]
        )
        print(f"\nQuery: {args.query!r}  (top {top_k}, threshold {settings.distance_threshold})")
        print("distance  keep?  user  source  page  text")
        for dist, meta, text in zip(result["distances"][0], result["metadatas"][0], result["documents"][0]):
            keep = "yes" if dist <= settings.distance_threshold else "NO"
            print(f"{dist:8.3f}  {keep:>5}  {meta.get('user_id')!s:>4}  {meta.get('source')}  p{meta.get('page')}  "
                  f"{_short(text, args.full, 70)}")


if __name__ == "__main__":
    main()
