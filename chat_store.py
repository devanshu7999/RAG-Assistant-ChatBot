"""
chat_store.py  -  Persistent per-user chat history storage
============================================================

Saves and loads chat history as JSON files on disk so that past
conversations survive across sessions, logouts, and server restarts.

Also handles guest session persistence and transfer to personal accounts.

Storage layout:
  data/chat_history/{user_id}.json      - per-user chat history
  data/chat_history/guest_{session}.json - guest session (temporary)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

# Directory for storing per-user chat history files
CHAT_HISTORY_DIR = os.path.join("data", "chat_history")


def _ensure_dir() -> None:
    """Create the chat history directory if it doesn't exist."""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


def _safe_filename(identifier: str) -> str:
    """Sanitise an identifier for safe filesystem use."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in identifier)


def _user_file(user_id: str) -> str:
    """Return the path to a user's chat history file."""
    return os.path.join(CHAT_HISTORY_DIR, f"{_safe_filename(user_id)}.json")


def _guest_file(guest_thread_id: str) -> str:
    """Return the path to a guest session's chat file."""
    return os.path.join(CHAT_HISTORY_DIR, f"{_safe_filename(guest_thread_id)}.json")


# Per-user chat history
def load_chat_history(user_id: str) -> List[Dict[str, Any]]:
    """
    Load a user's chat history from disk.
    Returns an empty list if no history exists yet.

    Each entry: {"thread_id": str, "name": str, "messages": [...]}
    """
    _ensure_dir()
    path = _user_file(user_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[chat_store.load] Failed to load history for {user_id}: {exc}")
        return []


def save_chat_history(user_id: str, history: List[Dict[str, Any]]) -> None:
    """
    Save a user's chat history to disk.
    Overwrites the existing file completely.
    """
    _ensure_dir()
    path = _user_file(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[chat_store.save] Failed to save history for {user_id}: {exc}")



# Guest session persistence
def save_guest_session(
    guest_thread_id: str,
    messages: List[Dict[str, str]],
) -> None:
    """
    Persist a guest's chat to disk so it survives page reloads
    and can be transferred on login.

    Stored as:
      {"session_id": "guest_abc123", "messages": [...]}
    """
    _ensure_dir()
    path = _guest_file(guest_thread_id)
    try:
        data = {
            "session_id": guest_thread_id,
            "messages": messages,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"[chat_store.save_guest] Failed to save guest session {guest_thread_id}: {exc}")


def load_guest_session(guest_thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a guest session from disk.
    Returns {"session_id": str, "messages": [...]} or None.
    """
    _ensure_dir()
    path = _guest_file(guest_thread_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def delete_guest_session(guest_thread_id: str) -> None:
    """Delete a guest session file after transfer."""
    _ensure_dir()
    path = _guest_file(guest_thread_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# Transfer: Guest → Personal Account
def transfer_guest_to_user(
    guest_thread_id: str,
    guest_messages: List[Dict[str, str]],
    user_id: str,
) -> bool:
    """
    Transfer a guest chat session into a user's personal account.

    Steps (mirrors the pattern from the implementation guide):
      1. Fetch guest chat (from session state / disk)
      2. Reassign ownership: attach to user_id
      3. Merge: append as a new past chat in user's history
      4. Clean up: delete the guest session file

    Returns True if transfer happened, False if nothing to transfer.
    """
    if not guest_messages:
        return False

    # Step 1 & 2: Create a chat entry owned by the user
    chat_entry = {
        "thread_id": guest_thread_id,  # keep original thread_id for traceability
        "name": "Guest Chat (transferred)",
        "messages": list(guest_messages),
    }

    # Step 3: Merge into user's existing chat history
    history = load_chat_history(user_id)
    # Check if already transferred (avoid duplicates on re-login)
    if any(c["thread_id"] == guest_thread_id for c in history):
        delete_guest_session(guest_thread_id)
        return False

    history.insert(0, chat_entry)  # newest first
    save_chat_history(user_id, history)

    # Step 4: Clean up guest session file
    delete_guest_session(guest_thread_id)

    return True
