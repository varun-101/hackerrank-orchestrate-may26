"""
Area Resolver — derives product_area from the corpus file path.

After Layer 2 retrieval, each RetrievedChunk carries a `product_area` field
populated from Path(source).parent.name (the directory immediately above the
.md file).  This module picks the best area from the reranked chunks and
provides it to the pipeline, falling back to the LLM-suggested value only
when retrieval confidence is too low or no chunks were returned.

Usage (in the pipeline, after reason() and before validate()):
    from area_resolver import resolve_product_area
    result.product_area = resolve_product_area(chunks, result.product_area)
"""

from __future__ import annotations

# Cross-encoder scores below this are treated as "not confident enough to
# trust the path-derived area".  The cross-encoder (ms-marco-MiniLM-L-6-v2)
# produces unbounded log-likelihood scores; empirically scores above -4.0
# indicate a reasonably relevant match.
CONFIDENCE_THRESHOLD: float = -4.0


def resolve_product_area(
    chunks,           # list[RetrievedChunk], sorted best-first (highest score first)
    llm_fallback: str,
    chunk_id: int = 0, # 1-based index from the LLM
) -> str:
    """
    Return the product_area for a ticket.

    Strategy:
      1. If the LLM returned a valid chunk_id (1-N) that grounded its answer,
         use that chunk's path-derived product_area.
      2. If chunk_id is missing/invalid, take the top-ranked chunk. If its score
         >= CONFIDENCE_THRESHOLD, use its product_area.
      3. Otherwise fall back to the LLM-generated value.

    Args:
        chunks:       Reranked list of RetrievedChunk (best first).
        llm_fallback: The product_area string produced by the LLM reasoner.
        chunk_id:     The 1-based index of the excerpt the LLM cited.

    Returns:
        A non-empty product_area string.
    """
    if chunks:
        # Strategy 1: Explicit LLM citation
        if 1 <= chunk_id <= len(chunks):
            cited_chunk = chunks[chunk_id - 1]
            if cited_chunk.product_area:
                return cited_chunk.product_area

        # Strategy 2: Cross-encoder confidence
        top = chunks[0]
        if top.score >= CONFIDENCE_THRESHOLD and top.product_area:
            return top.product_area

    return llm_fallback or ""
