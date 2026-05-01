"""
main.py — Pipeline entry point for HackerRank Orchestrate.

Reads support_tickets/support_tickets.csv, runs every ticket through the
full L1→L2→L3→L4 pipeline, and writes predictions to
support_tickets/output.csv.

Usage:
    cd code/
    python main.py                     # process all tickets
    python main.py --limit 5           # process first N tickets (dev mode)
    python main.py --verbose           # print per-ticket summaries
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Make sure the code/ directory is on sys.path when run from repo root
sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent.parent
INPUT_CSV     = ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV    = ROOT / "support_tickets" / "output.csv"

OUTPUT_FIELDS = [
    "issue", "subject", "company",
    "response", "product_area", "status", "request_type", "justification",
]

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _run_pipeline(issue: str, subject: str, company: str | None):
    """Run L1→L2→L3→L4 for a single ticket. Returns the final ReasonResult."""
    from preprocessor  import preprocess
    from retriever     import retrieve
    from reasoner      import reason
    from validator     import validate
    from area_resolver import resolve_product_area

    pre    = preprocess(issue, subject, company)
    text   = pre.english_text or pre.full_text
    chunks = retrieve(text, pre.domain)
    res    = reason(pre, chunks)

    # Override LLM-guessed product_area with the corpus-path-derived value.
    # The parent directory of the top-ranked .md file is the authoritative area.
    res.product_area = resolve_product_area(chunks, res.product_area, res.chunk_id)

    vr = validate(res, pre, chunks)
    return vr.result, chunks


def run(limit: int | None = None, verbose: bool = False) -> None:
    """Process all tickets and write output.csv."""
    from dotenv import load_dotenv
    load_dotenv()

    if not INPUT_CSV.exists():
        print(f"[main] ERROR: input CSV not found: {INPUT_CSV}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if limit:
        rows = rows[:limit]

    total   = len(rows)
    t_start = time.perf_counter()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        for idx, row in enumerate(rows, 1):
            issue   = row.get("Issue",   "").strip()
            subject = row.get("Subject", "").strip()
            company = row.get("Company", "").strip() or None

            if verbose:
                print(f"  [{idx}/{total}] {issue[:60]!r}…", end=" ", flush=True)

            t0 = time.perf_counter()
            try:
                result, chunks = _run_pipeline(issue, subject, company)
                elapsed = (time.perf_counter() - t0) * 1000

                writer.writerow({
                    "issue":         issue,
                    "subject":       subject,
                    "company":       company or "",
                    "response":      result.response,
                    "product_area":  result.product_area,
                    "status":        result.status,
                    "request_type":  result.request_type,
                    "justification": result.justification,
                })

                if verbose:
                    print(
                        f"status={result.status}  area={result.product_area!r}"
                        f"  rt={result.request_type}  ({elapsed:.0f}ms)"
                    )
                    print(f"chunks: {chunks}")


            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                if verbose:
                    print(f"ERROR ({elapsed:.0f}ms): {exc}")

                # Write a safe escalation so the row is never silently dropped
                writer.writerow({
                    "issue":         issue,
                    "subject":       subject,
                    "company":       company or "",
                    "response":      (
                        "Thank you for contacting support. Your request requires "
                        "review by a specialist. Our team will follow up shortly."
                    ),
                    "product_area":  (company or "").lower() or "general",
                    "status":        "escalated",
                    "request_type":  "product_issue",
                    "justification": f"Pipeline error: {exc}",
                })

    total_s = time.perf_counter() - t_start
    print(f"\n[main] Done. {total} tickets processed in {total_s:.1f}s")
    print(f"[main] Output written to: {OUTPUT_CSV}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the support-triage pipeline on support_tickets.csv"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N tickets (useful for testing)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-ticket status lines"
    )
    args = parser.parse_args()

    run(limit=args.limit, verbose=args.verbose)
