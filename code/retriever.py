"""
Layer 2 — Retrieval pipeline.

Three components as described in Architecture.md:

  1. Chunker    — splits corpus into ~512-token chunks (2048 chars) with
                  50-token overlap (200 chars), strips YAML frontmatter.
  2. Indexer    — embeds chunks offline with sentence-transformers, builds
                  a BM25 sparse index alongside; persists one .pkl shard
                  per domain under data/index/.
  3. Retriever  — at runtime: hybrid dense + BM25 retrieval over the
                  domain shard, then a cross-encoder reranker picks top 5.

Offline usage (run once before the pipeline):
    python build_index.py

Online usage (called by Layer 3):
    from retriever import retrieve
    chunks = retrieve(query="...", domain="HackerRank", top_k=5)
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

DOMAIN_DIRS: dict[str, Path] = {
    "HackerRank": DATA_DIR / "hackerrank",
    "Claude":     DATA_DIR / "claude",
    "Visa":       DATA_DIR / "visa",
}

# Chunking
CHUNK_CHARS    = 2048  # ≈ 512 tokens at ~4 chars/token
OVERLAP_CHARS  = 200   # ≈ 50 tokens

# Retrieval
DENSE_TOP_N    = 20    # dense candidates before reranking
BM25_TOP_N     = 20    # BM25 candidates before reranking
ALPHA          = 0.7   # hybrid weight: ALPHA * dense + (1-ALPHA) * bm25
RERANK_TOP_K   = 5     # final chunks returned after cross-encoder

# Model names (downloaded on first use from HuggingFace)
EMBED_MODEL    = "all-MiniLM-L6-v2"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text:     str
    source:   str   # path relative to DATA_DIR
    domain:   str
    chunk_id: int


@dataclass
class RetrievedChunk:
    text:   str
    source: str
    domain: str
    score:  float   # cross-encoder relevance score


# ---------------------------------------------------------------------------
# 1. Chunker
# ---------------------------------------------------------------------------
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _clean(text: str) -> str:
    """Strip YAML frontmatter and normalise whitespace."""
    text = _FRONTMATTER_RE.sub("", text, count=1)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping character windows.
    Tries to break at a newline near the window boundary to avoid
    cutting a sentence mid-word.
    """
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_CHARS, length)
        # Snap to nearest preceding newline to keep sentences whole
        if end < length:
            snap = text.rfind("\n", start, end)
            if snap > start + OVERLAP_CHARS:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - OVERLAP_CHARS
    return chunks


def _load_corpus_chunks(domain: str) -> list[Chunk]:
    """Read every .md file for a domain, clean and chunk it."""
    domain_dir = DOMAIN_DIRS.get(domain)
    if not domain_dir or not domain_dir.exists():
        return []

    chunks: list[Chunk] = []
    cid = 0
    for md_file in sorted(domain_dir.rglob("*.md")):
        raw = md_file.read_text(encoding="utf-8", errors="ignore")
        text = _clean(raw)
        if not text:
            continue
        source = str(md_file.relative_to(DATA_DIR))
        for chunk_text in _chunk_text(text):
            chunks.append(Chunk(text=chunk_text, source=source, domain=domain, chunk_id=cid))
            cid += 1
    return chunks


# ---------------------------------------------------------------------------
# 2. Indexer — model singletons + build_index()
# ---------------------------------------------------------------------------
_embed_model  = None
_rerank_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder(RERANK_MODEL)
    return _rerank_model


def build_index(domains: Optional[list[str]] = None, force: bool = False) -> None:
    """
    Build and persist domain-sharded retrieval indices.

    Each shard is saved as data/index/{domain_lower}.pkl and contains:
      - "chunks"     : list[Chunk]
      - "embeddings" : np.ndarray  shape (n, embed_dim), L2-normalised
      - "bm25"       : BM25Okapi object

    Args:
        domains: Domains to index. Defaults to all three.
        force:   Rebuild even if the index file already exists.
    """
    from rank_bm25 import BM25Okapi

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embed_model = _get_embed_model()
    targets = domains or list(DOMAIN_DIRS.keys())

    for domain in targets:
        index_path = INDEX_DIR / f"{domain.lower()}.pkl"
        if index_path.exists() and not force:
            print(f"[retriever] {domain}: index already exists, skipping (use --force to rebuild)")
            continue

        print(f"[retriever] {domain}: loading corpus …")
        chunks = _load_corpus_chunks(domain)
        if not chunks:
            print(f"[retriever] {domain}: WARNING — no documents found, skipping")
            continue

        texts = [c.text for c in chunks]
        print(f"[retriever] {domain}: embedding {len(chunks)} chunks …")
        embeddings = embed_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,   # cosine sim becomes a dot product
        )

        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)

        with open(index_path, "wb") as fh:
            pickle.dump({"chunks": chunks, "embeddings": embeddings, "bm25": bm25}, fh)

        print(f"[retriever] {domain}: {len(chunks)} chunks saved to {index_path}")


# ---------------------------------------------------------------------------
# 3. Retriever — hybrid retrieval + cross-encoder reranker
# ---------------------------------------------------------------------------
_index_cache: dict[str, dict] = {}


def _load_index(domain: str) -> Optional[dict]:
    if domain in _index_cache:
        return _index_cache[domain]
    path = INDEX_DIR / f"{domain.lower()}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    _index_cache[domain] = data
    return data


def _hybrid_candidates(query: str, index: dict) -> list[tuple[Chunk, float]]:
    """
    Dense + BM25 hybrid retrieval over one domain shard.

    Returns up to DENSE_TOP_N + BM25_TOP_N unique candidates ranked by
    the combined hybrid score.
    """
    chunks: list[Chunk]  = index["chunks"]
    embeddings: np.ndarray = index["embeddings"]
    bm25 = index["bm25"]

    # --- Dense (cosine similarity via dot product on normalised vectors) ---
    q_emb = _get_embed_model().encode([query], normalize_embeddings=True)[0]
    dense_scores: np.ndarray = embeddings @ q_emb          # shape (n,)

    # --- Sparse (BM25) ---
    tokens = query.lower().split()
    sparse_scores: np.ndarray = np.array(bm25.get_scores(tokens))

    # Normalise BM25 to [0, 1] so it's on the same scale as cosine similarity
    bm25_max = sparse_scores.max()
    if bm25_max > 0:
        sparse_scores = sparse_scores / bm25_max

    # --- Hybrid score and candidate selection ---
    hybrid = ALPHA * dense_scores + (1.0 - ALPHA) * sparse_scores

    # Take the union of top-N from each signal (avoids pure-dense or pure-sparse bias)
    dense_top  = set(np.argsort(dense_scores)[::-1][:DENSE_TOP_N].tolist())
    sparse_top = set(np.argsort(sparse_scores)[::-1][:BM25_TOP_N].tolist())
    candidate_indices = list(dense_top | sparse_top)

    return [(chunks[i], float(hybrid[i])) for i in candidate_indices]


def _rerank(query: str, candidates: list[tuple[Chunk, float]], top_k: int) -> list[RetrievedChunk]:
    """Score candidates with a cross-encoder and return the top_k."""
    reranker = _get_rerank_model()
    pairs  = [[query, c.text] for c, _ in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [
        RetrievedChunk(text=chunk.text, source=chunk.source, domain=chunk.domain, score=float(sc))
        for (chunk, _), sc in ranked[:top_k]
    ]


def retrieve(query: str, domain: str, top_k: int = RERANK_TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most relevant corpus chunks for a query.

    Args:
        query:  The (English-normalised) ticket text.
        domain: Domain shard to search.
                "ambiguous" searches all three shards.
        top_k:  Number of chunks after reranking.

    Returns:
        List of RetrievedChunk sorted by cross-encoder score (highest first).
        Returns an empty list if no index exists for the requested domain.
    """
    search_domains = list(DOMAIN_DIRS.keys()) if domain == "ambiguous" else [domain]

    all_candidates: list[tuple[Chunk, float]] = []
    for d in search_domains:
        idx = _load_index(d)
        if idx is None:
            print(f"[retriever] WARNING: no index found for {d!r} — run build_index.py first")
            continue
        all_candidates.extend(_hybrid_candidates(query, idx))

    if not all_candidates:
        return []

    return _rerank(query, all_candidates, top_k)
