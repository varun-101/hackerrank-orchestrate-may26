"""
Layer 5 — Observability.

Two components as described in Architecture.md:

  1. Tracer    — records per-stage timing and key decisions for every ticket.
                 Used by the pipeline (Layer 0) to wrap each layer call.
                 Produces a Trace object you can print or serialise.

  2. Evaluator — runs the full pipeline against sample_support_tickets.csv
                 (the ground-truth file) and prints a per-field accuracy report.
                 Run this after every prompt change to catch regressions before
                 you process the full support_tickets.csv.

Usage:
    # In the pipeline:
    from observability import Tracer
    tracer = Tracer(ticket_id=i, issue=issue)
    with tracer.stage("layer1"):
        pre = preprocess(issue, subject, company)
    tracer.record(domain=pre.domain, risk_score=pre.risk_score, ...)
    trace = tracer.finish(final_result)
    print(trace.summary())

    # Standalone evaluation:
    python observability.py
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

SAMPLE_CSV = Path(__file__).parent.parent / "support_tickets" / "sample_support_tickets.csv"

# ---------------------------------------------------------------------------
# 1.  Tracer — per-ticket timing + metadata
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """Full observability record for one processed ticket."""
    ticket_id:         int
    issue:             str
    timings_ms:        dict[str, float]
    total_ms:          float
    # Layer 1
    language:          str   = "en"
    domain:            str   = ""
    is_injection:      bool  = False
    risk_score:        float = 0.0
    risk_routed:       bool  = False
    # Layer 2
    chunks_retrieved:  int   = 0
    top_rerank_score:  float = 0.0
    # Layer 4
    validation_passed: bool  = True
    validation_issues: list[str] = field(default_factory=list)
    # Final output
    status:            str   = ""
    product_area:      str   = ""
    request_type:      str   = ""

    def summary(self) -> str:
        t = "  ".join(f"{k}={v:.0f}ms" for k, v in self.timings_ms.items())
        lines = [
            f"Ticket #{self.ticket_id} | total={self.total_ms:.0f}ms",
            f"  Timings   : {t}",
            f"  Pre-proc  : domain={self.domain!r}  lang={self.language}"
                f"  risk={self.risk_score:.2f}  injection={self.is_injection}"
                f"  risk_routed={self.risk_routed}",
            f"  Retrieval : {self.chunks_retrieved} chunks"
                f"  top_score={self.top_rerank_score:.3f}",
            f"  Output    : status={self.status!r}  area={self.product_area!r}"
                f"  type={self.request_type!r}",
        ]
        if self.validation_issues:
            lines.append(f"  Validation: FAILED — {'; '.join(self.validation_issues)}")
        else:
            lines.append(f"  Validation: passed")
        return "\n".join(lines)


class Tracer:
    """
    Lightweight per-ticket tracing.

    Wrap each pipeline stage with `tracer.stage(name)` and call
    `tracer.record(**kwargs)` to store any metadata.
    Finally call `tracer.finish(final_result)` to get the Trace.
    """

    def __init__(self, ticket_id: int, issue: str) -> None:
        self.ticket_id   = ticket_id
        self.issue       = issue[:120]
        self._timings:   dict[str, float] = {}
        self._meta:      dict             = {}
        self._wall_start = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Time a named pipeline stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = (time.perf_counter() - t0) * 1000

    def record(self, **kwargs) -> None:
        """Store arbitrary metadata (domain, risk_score, chunks_retrieved, …)."""
        self._meta.update(kwargs)

    def finish(self, final_result) -> Trace:
        total_ms = (time.perf_counter() - self._wall_start) * 1000
        return Trace(
            ticket_id  = self.ticket_id,
            issue      = self.issue,
            timings_ms = self._timings,
            total_ms   = total_ms,
            status       = getattr(final_result, "status",       ""),
            product_area = getattr(final_result, "product_area", ""),
            request_type = getattr(final_result, "request_type", ""),
            **self._meta,
        )


# ---------------------------------------------------------------------------
# 2.  Evaluator — accuracy report against sample_support_tickets.csv
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a","an","the","is","are","was","were","to","of","and","or","in","on",
    "at","for","with","from","by","as","that","this","it","be","been","have",
    "has","had","do","does","did","will","would","could","should","may","might",
    "can","your","our","their","its","my","you","we","they","i","not","no",
    "if","then","so","but","also","more","about","up","out","go","get","use",
    "any","all","each","how","what","when","where","who","which","please",
    "hi","hello","dear","thank","thanks","regarding","just","very","here",
}


def _tokenise(text: str) -> set[str]:
    return {
        w.lower() for w in re.findall(r"\b\w+\b", text)
        if w.lower() not in _STOPWORDS and len(w) > 2
    }


def _token_recall(pred: str, truth: str) -> float:
    """Fraction of ground-truth meaningful tokens that appear in the prediction."""
    truth_tok = _tokenise(truth)
    if not truth_tok:
        return 1.0
    return len(_tokenise(pred) & truth_tok) / len(truth_tok)


@dataclass
class _TicketEval:
    idx:           int
    company:       str
    status_ok:     bool
    rt_ok:         bool
    area_ok:       bool
    resp_recall:   float
    pred_status:   str
    pred_area:     str
    pred_rt:       str
    truth_status:  str
    truth_area:    str
    truth_rt:      str


def _run_pipeline(issue: str, subject: str, company: Optional[str]):
    """Run all four pipeline layers for one ticket."""
    sys.path.insert(0, str(Path(__file__).parent))

    from preprocessor   import preprocess
    from retriever      import retrieve
    from reasoner       import reason
    from validator      import validate
    from area_resolver  import resolve_product_area
    import dataclasses
    import json

    pre    = preprocess(issue, subject, company)
    text   = pre.english_text or pre.full_text
    chunks = retrieve(text, pre.domain)
    res    = reason(pre, chunks)
    print(f"chunks : {chunks}")
    print(f"\n[LLM JSON] {json.dumps(dataclasses.asdict(res), indent=2)}")

    # Override LLM-guessed product_area with the corpus-path-derived value.
    # The parent directory of the top-ranked .md file is the authoritative area.
    res.product_area = resolve_product_area(chunks, res.product_area, res.chunk_id)

    vr = validate(res, pre, chunks)
    return vr.result


def evaluate_on_sample(verbose: bool = True) -> dict[str, float]:
    """
    Run the full pipeline on every row of sample_support_tickets.csv,
    compare against ground truth, and print a per-field accuracy report.

    Returns a dict: {"status": 0.9, "request_type": 1.0, "product_area": 0.7,
                     "response_recall": 0.74}
    """
    if not SAMPLE_CSV.exists():
        print(f"[eval] sample CSV not found: {SAMPLE_CSV}")
        return {}

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("[eval] DEEPSEEK_API_KEY not set — LLM layers will fail gracefully.")
        print("       Set the key to get meaningful eval results.\n")

    rows: list[dict] = []
    with open(SAMPLE_CSV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    evals: list[_TicketEval] = []

    for idx, row in enumerate(rows, 1):
        issue   = row.get("Issue",   "").strip()
        subject = row.get("Subject", "").strip()
        company = row.get("Company", "").strip() or None

        truth_status = row.get("Status",       "").strip().lower()
        truth_area   = row.get("Product Area", "").strip().lower()
        truth_rt     = row.get("Request Type", "").strip().lower()
        truth_resp   = row.get("Response",     "").strip()

        if verbose:
            print(f"  [{idx}/{len(rows)}] {issue[:60]!r} …", end=" ", flush=True)

        try:
            result = _run_pipeline(issue, subject, company)
            pred_status = result.status.lower().strip()
            pred_area   = result.product_area.lower().strip()
            pred_rt     = result.request_type.lower().strip()
            pred_resp   = result.response.strip()
        except Exception as exc:
            if verbose:
                print(f"ERROR: {exc}")
            evals.append(_TicketEval(
                idx=idx, company=company or "None",
                status_ok=False, rt_ok=False, area_ok=False, resp_recall=0.0,
                pred_status="error", pred_area="error", pred_rt="error",
                truth_status=truth_status, truth_area=truth_area, truth_rt=truth_rt,
            ))
            continue

        status_ok   = pred_status == truth_status
        rt_ok       = pred_rt     == truth_rt
        area_ok     = pred_area   == truth_area
        resp_recall = _token_recall(pred_resp, truth_resp)

        evals.append(_TicketEval(
            idx=idx, company=company or "None",
            status_ok=status_ok, rt_ok=rt_ok, area_ok=area_ok, resp_recall=resp_recall,
            pred_status=pred_status, pred_area=pred_area, pred_rt=pred_rt,
            truth_status=truth_status, truth_area=truth_area, truth_rt=truth_rt,
        ))

        if verbose:
            marks = (
                ("status",  "ok" if status_ok else "FAIL"),
                ("rt",      "ok" if rt_ok     else "FAIL"),
                ("area",    "ok" if area_ok   else "FAIL"),
            )
            print("  ".join(f"{k}={m}" for k, m in marks)
                  + f"  resp={resp_recall:.0%}")

    if not evals:
        return {}

    n = len(evals)
    status_acc  = sum(e.status_ok   for e in evals) / n
    rt_acc      = sum(e.rt_ok       for e in evals) / n
    area_acc    = sum(e.area_ok     for e in evals) / n
    resp_recall = sum(e.resp_recall for e in evals) / n
    overall     = (status_acc + rt_acc + area_acc) / 3

    # ---- Print report ----
    w = 44
    print()
    print("=" * w)
    print(f"  EVALUATION REPORT  ({n} sample tickets)")
    print("=" * w)
    print(f"  {'Field':<18}  {'Correct':>7}  {'Total':>5}  {'Acc':>7}")
    print(f"  {'-'*18}  {'-'*7}  {'-'*5}  {'-'*7}")

    for label, correct_count, acc in [
        ("status",        int(status_acc  * n), status_acc),
        ("request_type",  int(rt_acc      * n), rt_acc),
        ("product_area",  int(area_acc    * n), area_acc),
    ]:
        print(f"  {label:<18}  {correct_count:>7}  {n:>5}  {acc:>6.1%}")

    print(f"  {'response (recall)':<18}  {'—':>7}  {'—':>5}  {resp_recall:>6.1%}")
    print(f"  {'-'*18}  {'-'*7}  {'-'*5}  {'-'*7}")
    print(f"  {'overall (3 fields)':<18}  {'':>7}  {'':>5}  {overall:>6.1%}")
    print("=" * w)

    # ---- Per-ticket mismatches ----
    mismatches = [e for e in evals if not (e.status_ok and e.rt_ok and e.area_ok)]
    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)} tickets):")
        for e in mismatches:
            parts = []
            if not e.status_ok:
                parts.append(f"status: got={e.pred_status!r} exp={e.truth_status!r}")
            if not e.rt_ok:
                parts.append(f"rt: got={e.pred_rt!r} exp={e.truth_rt!r}")
            if not e.area_ok:
                parts.append(f"area: got={e.pred_area!r} exp={e.truth_area!r}")
            print(f"    #{e.idx} [{e.company}]  " + "  |  ".join(parts))

    print()
    return {
        "status":          status_acc,
        "request_type":    rt_acc,
        "product_area":    area_acc,
        "response_recall": resp_recall,
        "overall":         overall,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent))
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate pipeline on sample tickets")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-ticket lines")
    args = parser.parse_args()

    evaluate_on_sample(verbose=not args.quiet)
