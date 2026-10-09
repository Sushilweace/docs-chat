"""Retrieval evaluation: reports hit rate (and MRR) over a labelled question set.

Usage (from the project root, with the venv active and documents already
uploaded + ingested for the chosen user)::

    python eval/run_eval.py --user-email you@example.com
    python eval/run_eval.py --user-email you@example.com --top-k 3 --no-threshold

``eval/questions.json`` is a list of objects::

    {"question": "...", "expected_source": "file.pdf", "expected_page": 4, "contains": "optional text"}

``expected_page`` and ``contains`` are optional extra constraints. A question
with ``"expected_source": null`` is a *negative* question: the correct outcome
is that retrieval returns nothing (so the app answers "not in the documents").
Only the retrieval layer is evaluated - no LLM calls are made.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # allow `python eval/run_eval.py` from anywhere

from sqlalchemy import select  # noqa: E402

from app.core.database import SessionLocal  # noqa: E402
from app.models.db import User  # noqa: E402
from app.services.retrieval import RetrievedChunk, retrieve  # noqa: E402


@dataclass(frozen=True)
class Case:
    """One labelled evaluation question."""

    question: str
    expected_source: str | None
    expected_page: int | None = None
    contains: str | None = None


def load_cases(path: Path) -> list[Case]:
    """Load and validate evaluation cases from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"{path} must contain a non-empty JSON list")
    cases = []
    for i, item in enumerate(raw, start=1):
        if "question" not in item or "expected_source" not in item:
            raise SystemExit(f"{path} item {i}: 'question' and 'expected_source' are required")
        cases.append(
            Case(
                question=item["question"],
                expected_source=item["expected_source"],
                expected_page=item.get("expected_page"),
                contains=item.get("contains"),
            )
        )
    return cases


def matches(case: Case, chunk: RetrievedChunk) -> bool:
    """True if ``chunk`` satisfies every constraint the case specifies."""
    if chunk.source != case.expected_source:
        return False
    if case.expected_page is not None and chunk.page != case.expected_page:
        return False
    if case.contains and case.contains.lower() not in chunk.text.lower():
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit rate.")
    parser.add_argument("--user-email", required=True, help="Owner of the ingested documents")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    parser.add_argument("--top-k", type=int, default=None, help="Override TOP_K")
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Ignore DISTANCE_THRESHOLD to measure pure ranking quality",
    )
    args = parser.parse_args()

    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.email == args.user_email.strip().lower()))
    if user is None:
        print(f"No user with email {args.user_email!r}. Sign up and upload documents first.")
        return 2

    cases = load_cases(args.questions)
    threshold = 2.0 if args.no_threshold else None  # cosine distance never exceeds 2

    positives = hits = negatives = refusals = 0
    reciprocal_ranks: list[float] = []

    print(f"{'result':<8} {'rank':<5} {'best dist':<10} question")
    print("-" * 78)
    for case in cases:
        chunks = retrieve(case.question, user.id, top_k=args.top_k, distance_threshold=threshold)
        best = f"{chunks[0].distance:.3f}" if chunks else "-"

        if case.expected_source is None:  # negative question
            negatives += 1
            ok = not chunks
            refusals += ok
            print(f"{'OK' if ok else 'MISS':<8} {'n/a':<5} {best:<10} [negative] {case.question}")
            continue

        positives += 1
        rank = next((i for i, c in enumerate(chunks, 1) if matches(case, c)), None)
        hits += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        print(f"{'HIT' if rank else 'MISS':<8} {rank or '-':<5} {best:<10} {case.question}")

    print("-" * 78)
    if positives:
        print(f"Retrieval hit rate : {hits}/{positives} = {hits / positives:.1%}")
        print(f"MRR                : {sum(reciprocal_ranks) / positives:.3f}")
    if negatives:
        print(f"Correct refusals   : {refusals}/{negatives} = {refusals / negatives:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
