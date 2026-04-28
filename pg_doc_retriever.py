"""
pg_doc_retriever.py  –  Persistent full-document retrieval fallback (per-user)
===============================================================================

Layer 8 (fallback): When the Vector Store Retriever (semantic search over
*past conversations*) does not yield a confident answer, this module searches
the user's persistent full-document chunks stored in PostgreSQL.

This is the LAST retrieval layer before a genuine "I don't know" response.

Fallback chain:
  Buffer Window (last 10 turns)
    → Global Summary Memory + Account KG + User Memory + User KG
      → Vector Store Retriever  (semantic search over closed past conversations)
        → [THIS LAYER]  PostgreSQL document chunks  (full-document storage, per-user)

Key design decisions
--------------------
* Per-user isolation: every query is scoped to `user_id` (WHERE user_id = %s).
  No cross-user data leakage is possible.
* Semantic ranking: uses SentenceTransformer cosine similarity to rank chunks
  from PostgreSQL so only relevant passages are injected.
* Threshold-gated: only injects chunks whose cosine similarity exceeds
  PG_DOC_RELEVANCE_THRESHOLD (default 0.25), preventing noise injection.
* Triggered conditionally: RAGEngine / MemoryManager only calls this when
  the vector store returned an empty or low-confidence block.
* Guest users are never given access (no PG rows, no personal docs).
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer

from app_config import DATABASE_URL

# ── Tunable constants ──────────────────────────────────────────────────────────
# Minimum cosine similarity for a PostgreSQL chunk to be injected as context.
# Raise this to be more selective; lower it to be more permissive.
PG_DOC_RELEVANCE_THRESHOLD = 0.25

# Maximum chunks to retrieve from PostgreSQL per query (before threshold filter).
PG_DOC_TOP_K = 5

# Maximum chunks to actually inject into the prompt after threshold filtering.
PG_DOC_INJECT_TOP_N = 3


def _connect() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


class PgDocRetriever:
    """
    Semantic retrieval over a user's persistent document chunks in PostgreSQL.

    Usage
    -----
    Call ``retrieve(user_id, query, st_model)`` — it returns a formatted
    prompt block (or an empty string if nothing relevant was found).

    The caller (MemoryManager.build_context) decides *when* to invoke this:
    only when the Vector Store Retriever block is empty or flagged as low-
    confidence.
    """

    # ── Public interface ──────────────────────────────────────────────────────

    @staticmethod
    def retrieve(
        user_id:  str,
        query:    str,
        st_model: SentenceTransformer,
        top_k:    int = PG_DOC_TOP_K,
        top_n:    int = PG_DOC_INJECT_TOP_N,
        threshold: float = PG_DOC_RELEVANCE_THRESHOLD,
    ) -> str:
        """
        Search the user's PostgreSQL document chunks for passages relevant
        to *query*.  Returns a formatted block for prompt injection, or ""
        if nothing relevant was found.

        Parameters
        ----------
        user_id  : authenticated user — guests must never call this.
        query    : the current user question.
        st_model : shared SentenceTransformer instance (reuse, don't reload).
        top_k    : candidate chunks to fetch from Postgres before re-ranking.
        top_n    : max chunks to inject after threshold filtering.
        threshold: cosine similarity floor (0-1).  Chunks below this are dropped.
        """
        if not user_id or user_id.startswith("guest_"):
            return ""

        # ── Load candidate chunks from PostgreSQL ─────────────────────────────
        # We fetch top_k candidates using a simple text-match heuristic
        # (keyword tokens) and then re-rank by embedding similarity.
        # This avoids loading ALL of the user's chunks into memory.
        candidates = PgDocRetriever._fetch_candidates(user_id, query, top_k)
        if not candidates:
            return ""

        # ── Re-rank by cosine similarity ──────────────────────────────────────
        ranked = PgDocRetriever._rank_by_similarity(query, candidates, st_model)

        # ── Filter by threshold and take top_n ────────────────────────────────
        relevant = [
            (chunk, score, doc_name, page)
            for chunk, score, doc_name, page in ranked
            if score >= threshold
        ][:top_n]

        if not relevant:
            return ""

        # ── Format the prompt block ───────────────────────────────────────────
        lines = ["[Document Knowledge (persistent storage)]"]
        for i, (chunk, score, doc_name, page) in enumerate(relevant, 1):
            label = f"{doc_name} — Page {page}" if page is not None else doc_name
            truncated = chunk[:500] + "…" if len(chunk) > 500 else chunk
            lines.append(f"  {i}. [{label}] {truncated}")

        return "\n".join(lines)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fetch_candidates(
        user_id: str,
        query:   str,
        top_k:   int,
    ) -> List[dict]:
        """
        Fetch candidate chunks from PostgreSQL.

        Strategy: use PostgreSQL full-text search (to_tsvector / plainto_tsquery)
        for a lightweight first-pass ranking.  Falls back to returning the
        most recent `top_k * 3` chunks if full-text yields nothing (e.g. very
        short or non-English queries).

        All rows are strictly scoped to `user_id`.
        """
        # Extract meaningful tokens (> 3 chars) for the FTS query
        tokens = [w for w in query.lower().split() if len(w) > 3]
        fts_query = " | ".join(tokens) if tokens else ""

        try:
            with _connect() as conn:
                if fts_query:
                    rows = conn.execute(
                        """
                        SELECT chunk_text, doc_name, page_number,
                               ts_rank(to_tsvector('english', chunk_text),
                                       plainto_tsquery('english', %s)) AS fts_rank
                        FROM user_document_chunks
                        WHERE user_id = %s
                          AND to_tsvector('english', chunk_text)
                              @@ plainto_tsquery('english', %s)
                        ORDER BY fts_rank DESC
                        LIMIT %s
                        """,
                        (fts_query, user_id, fts_query, top_k * 3),
                    ).fetchall()

                    if rows:
                        return [dict(r) for r in rows]

                # FTS returned nothing — fall back to recency (latest docs first)
                rows = conn.execute(
                    """
                    SELECT chunk_text, doc_name, page_number
                    FROM user_document_chunks
                    WHERE user_id = %s
                    ORDER BY upload_order DESC, chunk_index ASC
                    LIMIT %s
                    """,
                    (user_id, top_k * 3),
                ).fetchall()
                return [dict(r) for r in rows]

        except Exception as exc:
            print(f"[PgDocRetriever._fetch_candidates] user={user_id!r}: {exc}")
            return []

    @staticmethod
    def _rank_by_similarity(
        query:      str,
        candidates: List[dict],
        st_model:   SentenceTransformer,
    ) -> List[Tuple[str, float, str, Optional[int]]]:
        """
        Re-rank candidate chunks by cosine similarity to the query embedding.

        Returns list of (chunk_text, score, doc_name, page_number) sorted
        descending by score.
        """
        texts = [r["chunk_text"] for r in candidates]
        query_emb  = st_model.encode(query,  normalize_embeddings=True)
        chunk_embs = st_model.encode(texts,  normalize_embeddings=True,
                                     show_progress_bar=False)

        scores = np.dot(chunk_embs, query_emb).tolist()

        ranked = sorted(
            zip(texts, scores,
                [r["doc_name"] for r in candidates],
                [r.get("page_number") for r in candidates]),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    # ── Utility: does this user have any stored documents? ────────────────────

    @staticmethod
    def has_documents(user_id: str) -> bool:
        """
        Return True if the user has at least one chunk in PostgreSQL.
        Cheap EXISTS query — used by the caller to skip retrieval entirely
        for users who have never uploaded a document.
        """
        if not user_id or user_id.startswith("guest_"):
            return False
        try:
            with _connect() as conn:
                row = conn.execute(
                    "SELECT 1 FROM user_document_chunks WHERE user_id = %s LIMIT 1",
                    (user_id,),
                ).fetchone()
            return row is not None
        except Exception as exc:
            print(f"[PgDocRetriever.has_documents] user={user_id!r}: {exc}")
            return False

    @staticmethod
    def list_doc_names(user_id: str) -> List[str]:
        """Return distinct document names stored for this user."""
        if not user_id or user_id.startswith("guest_"):
            return []
        try:
            with _connect() as conn:
                rows = conn.execute(
                    """
                    SELECT DISTINCT doc_name
                    FROM user_document_chunks
                    WHERE user_id = %s
                    ORDER BY doc_name
                    """,
                    (user_id,),
                ).fetchall()
            return [r["doc_name"] for r in rows]
        except Exception as exc:
            print(f"[PgDocRetriever.list_doc_names] user={user_id!r}: {exc}")
            return []
