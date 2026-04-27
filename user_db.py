"""
user_db.py  -  PostgreSQL-backed user account management
=========================================================

Handles signup, login verification, and per-user record tracking.

Schema
------
  users
  ─────
  id            SERIAL PRIMARY KEY
  user_id       TEXT UNIQUE NOT NULL        ← the app-level username / key
  email         TEXT UNIQUE NOT NULL
  display_name  TEXT NOT NULL
  password_hash TEXT NOT NULL               ← bcrypt hash (via streamlit-authenticator)
  created_at    TIMESTAMPTZ DEFAULT now()
  last_login    TIMESTAMPTZ
  is_active     BOOLEAN DEFAULT TRUE

  user_memory_facts
  ─────────────────
  id          SERIAL PRIMARY KEY
  user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE
  fact_id     TEXT NOT NULL
  text        TEXT NOT NULL
  source      TEXT NOT NULL DEFAULT 'auto'
  created_at  TIMESTAMPTZ DEFAULT now()
  UNIQUE(user_id, fact_id)

  user_conversations
  ──────────────────
  id          SERIAL PRIMARY KEY
  user_id     TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE
  thread_id   TEXT NOT NULL
  title       TEXT
  created_at  TIMESTAMPTZ DEFAULT now()
  UNIQUE(user_id, thread_id)

  user_document_chunks
  ────────────────────
  id            SERIAL PRIMARY KEY
  user_id       TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE
  doc_name      TEXT NOT NULL
  chunk_index   INT  NOT NULL
  chunk_text    TEXT NOT NULL
  page_number   INT
  upload_order  INT  NOT NULL DEFAULT 1
  created_at    TIMESTAMPTZ DEFAULT now()
  UNIQUE(user_id, doc_name, chunk_index)

Isolation guarantee
-------------------
Every SELECT/INSERT/UPDATE/DELETE in this module is parameterised with
user_id, so one user can never read or write another's rows.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from config import DATABASE_URL


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

def _connect() -> psycopg.Connection:
    """Open a new synchronous psycopg3 connection."""
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


# ─────────────────────────────────────────────────────────────────────────────
# Schema bootstrap  (idempotent — safe to call on every startup)
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT        UNIQUE NOT NULL,
    email         TEXT        UNIQUE NOT NULL,
    display_name  TEXT        NOT NULL,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login    TIMESTAMPTZ,
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS user_memory_facts (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    fact_id    TEXT        NOT NULL,
    text       TEXT        NOT NULL,
    source     TEXT        NOT NULL DEFAULT 'auto',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, fact_id)
);

CREATE TABLE IF NOT EXISTS user_conversations (
    id         SERIAL PRIMARY KEY,
    user_id    TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    thread_id  TEXT        NOT NULL,
    title      TEXT,
    summary    TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, thread_id)
);

CREATE TABLE IF NOT EXISTS user_document_chunks (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    doc_name      TEXT        NOT NULL,
    chunk_index   INT         NOT NULL,
    chunk_text    TEXT        NOT NULL,
    page_number   INT,
    upload_order  INT         NOT NULL DEFAULT 1,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, doc_name, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_umf_user_id ON user_memory_facts (user_id);
CREATE INDEX IF NOT EXISTS idx_uc_user_id  ON user_conversations (user_id);
CREATE INDEX IF NOT EXISTS idx_udc_user_id ON user_document_chunks (user_id);
"""


def init_db() -> None:
    """
    Create all tables and indexes if they don't exist yet.
    Call once at application startup (e.g. inside RAGEngine.__init__).
    """
    try:
        with _connect() as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()
        print("[user_db] Schema initialised.")
    except Exception as exc:
        print(f"[user_db.init_db] Failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# User account operations
# ─────────────────────────────────────────────────────────────────────────────

def user_exists(user_id: str) -> bool:
    """Return True if a user with this user_id already exists."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = %s", (user_id,)
            ).fetchone()
        return row is not None
    except Exception as exc:
        print(f"[user_db.user_exists] {exc}")
        return False


def email_exists(email: str) -> bool:
    """Return True if this email is already registered."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE email = %s", (email.lower().strip(),)
            ).fetchone()
        return row is not None
    except Exception as exc:
        print(f"[user_db.email_exists] {exc}")
        return False


def create_user(
    user_id:      str,
    email:        str,
    display_name: str,
    password_hash: str,
) -> bool:
    """
    Insert a new user record.

    password_hash must already be bcrypt-hashed (e.g. via
    streamlit_authenticator.Hasher.hash()).

    Returns True on success, False if user_id or email already exists.
    """
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, email, display_name, password_hash)
                VALUES (%s, %s, %s, %s)
                """,
                (user_id.strip(), email.lower().strip(), display_name.strip(), password_hash),
            )
            conn.commit()
        print(f"[user_db] Created user: {user_id!r}")
        return True
    except psycopg.errors.UniqueViolation:
        print(f"[user_db.create_user] Duplicate user_id or email: {user_id!r}")
        return False
    except Exception as exc:
        print(f"[user_db.create_user] {exc}")
        return False


def get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a user record by user_id.
    Returns a dict or None if not found.
    """
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = %s AND is_active = TRUE",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[user_db.get_user] {exc}")
        return None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch a user record by email address."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = %s AND is_active = TRUE",
                (email.lower().strip(),),
            ).fetchone()
        return dict(row) if row else None
    except Exception as exc:
        print(f"[user_db.get_user_by_email] {exc}")
        return None


def update_last_login(user_id: str) -> None:
    """Stamp last_login for the given user (called after successful authentication)."""
    try:
        with _connect() as conn:
            conn.execute(
                "UPDATE users SET last_login = now() WHERE user_id = %s",
                (user_id,),
            )
            conn.commit()
    except Exception as exc:
        print(f"[user_db.update_last_login] {exc}")


def get_all_users_for_auth_yaml() -> List[Dict[str, Any]]:
    """
    Return all active users in the shape expected by streamlit-authenticator
    so auth.yaml can be rebuilt dynamically from the DB.

    Returns list of dicts:
      [{"user_id": str, "email": str, "display_name": str, "password_hash": str}, ...]
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT user_id, email, display_name, password_hash "
                "FROM users WHERE is_active = TRUE ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[user_db.get_all_users_for_auth_yaml] {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Per-user memory facts  (Postgres mirror of in-memory UserMemory)
# ─────────────────────────────────────────────────────────────────────────────

def save_memory_fact(user_id: str, fact_id: str, text: str, source: str = "auto") -> None:
    """
    Upsert a memory fact for *this user only*.
    Another user's facts are in separate rows keyed by their own user_id.
    """
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memory_facts (user_id, fact_id, text, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, fact_id) DO UPDATE
                    SET text = EXCLUDED.text, source = EXCLUDED.source
                """,
                (user_id, fact_id, text, source),
            )
            conn.commit()
    except Exception as exc:
        print(f"[user_db.save_memory_fact] user={user_id!r}: {exc}")


def load_memory_facts(user_id: str) -> List[Dict[str, Any]]:
    """
    Load all memory facts for *this user only*.
    The WHERE user_id = %s clause ensures strict isolation.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT fact_id, text, source, created_at "
                "FROM user_memory_facts WHERE user_id = %s ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[user_db.load_memory_facts] user={user_id!r}: {exc}")
        return []


def delete_memory_fact(user_id: str, fact_id: str) -> bool:
    """Delete a specific fact belonging to *this user only*."""
    try:
        with _connect() as conn:
            result = conn.execute(
                "DELETE FROM user_memory_facts WHERE user_id = %s AND fact_id = %s",
                (user_id, fact_id),
            )
            conn.commit()
        return (result.rowcount or 0) > 0
    except Exception as exc:
        print(f"[user_db.delete_memory_fact] user={user_id!r}: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-user conversation index
# ─────────────────────────────────────────────────────────────────────────────

def save_conversation(
    user_id: str,
    thread_id: str,
    title: str = "",
    summary: str = "",
) -> None:
    """
    Upsert a conversation record for *this user only*.
    The (user_id, thread_id) UNIQUE constraint prevents cross-user collisions.
    """
    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO user_conversations (user_id, thread_id, title, summary)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, thread_id) DO UPDATE
                    SET title = EXCLUDED.title, summary = EXCLUDED.summary
                """,
                (user_id, thread_id, title, summary),
            )
            conn.commit()
    except Exception as exc:
        print(f"[user_db.save_conversation] user={user_id!r}: {exc}")


def load_conversations(user_id: str) -> List[Dict[str, Any]]:
    """
    Load all conversation records for *this user only*.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT thread_id, title, summary, created_at "
                "FROM user_conversations WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[user_db.load_conversations] user={user_id!r}: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Per-user document chunks  (PostgreSQL = source of truth for PDF persistence)
# ─────────────────────────────────────────────────────────────────────────────

def save_document_chunks(
    user_id:      str,
    doc_name:     str,
    chunks:       List[Dict[str, Any]],
    upload_order: int = 1,
) -> None:
    """
    Batch-insert PDF chunks for *this user only*.

    Each element in `chunks` is a dict with keys:
        - text:        str   (the chunk content)
        - page_number: int | None
        - chunk_index: int   (sequential position within the document)

    Uses ON CONFLICT to handle re-uploads of the same document gracefully.
    """
    if not chunks:
        return
    try:
        with _connect() as conn:
            for ch in chunks:
                conn.execute(
                    """
                    INSERT INTO user_document_chunks
                        (user_id, doc_name, chunk_index, chunk_text, page_number, upload_order)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, doc_name, chunk_index) DO UPDATE
                        SET chunk_text    = EXCLUDED.chunk_text,
                            page_number   = EXCLUDED.page_number,
                            upload_order  = EXCLUDED.upload_order
                    """,
                    (
                        user_id,
                        doc_name,
                        ch["chunk_index"],
                        ch["text"],
                        ch.get("page_number"),
                        upload_order,
                    ),
                )
            conn.commit()
        print(f"[user_db] Saved {len(chunks)} chunks for user={user_id!r}, doc={doc_name!r}")
    except Exception as exc:
        print(f"[user_db.save_document_chunks] user={user_id!r}: {exc}")


def load_document_chunks(user_id: str) -> List[Dict[str, Any]]:
    """
    Load ALL document chunks belonging to *this user only*.

    Returns a list of dicts:
        [{"doc_name": str, "chunk_index": int, "chunk_text": str,
          "page_number": int|None, "upload_order": int}, ...]

    The WHERE user_id = %s clause guarantees strict isolation.
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_name, chunk_index, chunk_text, page_number, upload_order
                FROM user_document_chunks
                WHERE user_id = %s
                ORDER BY upload_order, chunk_index
                """,
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[user_db.load_document_chunks] user={user_id!r}: {exc}")
        return []


def delete_document_chunks(user_id: str, doc_name: Optional[str] = None) -> int:
    """
    Delete chunks for *this user only*.

    If doc_name is provided, delete only that document's chunks.
    If doc_name is None, delete ALL chunks for the user.

    Returns the number of rows deleted.
    """
    try:
        with _connect() as conn:
            if doc_name:
                result = conn.execute(
                    "DELETE FROM user_document_chunks WHERE user_id = %s AND doc_name = %s",
                    (user_id, doc_name),
                )
            else:
                result = conn.execute(
                    "DELETE FROM user_document_chunks WHERE user_id = %s",
                    (user_id,),
                )
            conn.commit()
        deleted = result.rowcount or 0
        print(f"[user_db] Deleted {deleted} chunks for user={user_id!r}, doc={doc_name!r}")
        return deleted
    except Exception as exc:
        print(f"[user_db.delete_document_chunks] user={user_id!r}: {exc}")
        return 0


def get_user_doc_names(user_id: str) -> List[Dict[str, Any]]:
    """
    Return distinct document names and their upload_order for *this user only*.

    Returns: [{"doc_name": str, "upload_order": int, "chunk_count": int}, ...]
    """
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_name, MAX(upload_order) AS upload_order, COUNT(*) AS chunk_count
                FROM user_document_chunks
                WHERE user_id = %s
                GROUP BY doc_name
                ORDER BY upload_order
                """,
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        print(f"[user_db.get_user_doc_names] user={user_id!r}: {exc}")
        return []
