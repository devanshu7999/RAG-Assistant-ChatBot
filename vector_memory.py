"""
vector_memory.py  –  Semantic long-term memory retrieval (per-user)
====================================================================

Each logged-in user gets a dedicated ChromaDB collection that stores
embeddings of past conversation exchanges.  When the user starts a
new chat, their closed conversation is vectorised and stored here.

On each query, semantically similar past interactions are retrieved
and injected as context — enabling the assistant to recall relevant
information from conversations that have long left the buffer window.

Guest users do NOT have access to this layer.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from app_config import VECTOR_MEMORY_TOP_K


class VectorMemoryStore:
    """
    ChromaDB-backed semantic memory for logged-in users.

    Each user gets their own collection: ``user_memory_{user_id}``
    Conversation pairs are embedded as single documents when a chat is closed.
    """

    def __init__(
        self,
        chroma_client: chromadb.ClientAPI,
        st_model: SentenceTransformer,
    ):
        self._chroma = chroma_client
        self._model  = st_model
        # cache of per-user collections
        self._collections: Dict[str, chromadb.Collection] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        """
        Raise ValueError for blank or guest user_ids.

        Guest users must never access the vector store — they have
        no persistent identity and no right to read any user's memories.
        """
        if not user_id or not user_id.strip():
            raise ValueError("[VectorMemoryStore] user_id must not be empty.")
        if user_id.startswith("guest_"):
            raise ValueError(
                f"[VectorMemoryStore] Guest users are not permitted to access "
                f"vector memory (user_id={user_id!r})."
            )

    def _get_collection(self, user_id: str) -> chromadb.Collection:
        """
        Get or create the user-specific vector memory collection.

        Each user gets a *dedicated, isolated* collection whose name is
        derived deterministically from their user_id.  No cross-user
        access is possible because every query/store call goes through
        this method which enforces the user_id binding.
        """
        self._validate_user_id(user_id)
        if user_id not in self._collections:
            # Sanitise user_id for ChromaDB collection name rules:
            # 3-63 chars, alphanumeric + underscores/hyphens, no leading/trailing hyphens
            safe_uid = "".join(
                c if c.isalnum() or c in ("_", "-") else "_"
                for c in user_id
            )
            collection_name = f"umem_{safe_uid}"[:63]
            self._collections[user_id] = self._chroma.get_or_create_collection(
                name=collection_name,
                metadata={
                    "owner_user_id": user_id,   # recorded for audit / cleanup
                    "description": f"Isolated long-term memory for user: {user_id}",
                },
            )
        return self._collections[user_id]

    # ── Store conversation ────────────────────────────────────────────────────

    def store_conversation(
        self,
        user_id:   str,
        thread_id: str,
        pairs:     List[Tuple[str, str]],   # [(human, assistant), …]
    ) -> int:
        """
        Embed and store conversation pairs into the user's vector memory.

        Each pair is stored as a separate document so retrieval can match
        at the exchange level (more granular than whole-conversation).

        Returns the number of documents stored.
        """
        if not pairs:
            return 0

        # Raises if user_id is blank or a guest — enforced before any DB access
        self._validate_user_id(user_id)
        collection = self._get_collection(user_id)

        documents:  List[str]            = []
        metadatas:  List[Dict[str, Any]] = []
        ids:        List[str]            = []

        for i, (human, assistant) in enumerate(pairs):
            # Combine the exchange into a single document for embedding
            doc_text = f"User: {human}\nAssistant: {assistant}"
            doc_id   = f"{thread_id}_{i}_{uuid.uuid4().hex[:6]}"

            documents.append(doc_text)
            metadatas.append({
                "thread_id": thread_id,
                "user_id":   user_id,
                "turn":      i,
                "user_msg":  human[:500],   # store truncated for metadata queries
            })
            ids.append(doc_id)

        # Embed and add in one batch
        embeddings = self._model.encode(
            documents, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return len(documents)

    # ── Retrieve similar past interactions ────────────────────────────────────

    def retrieve(
        self,
        user_id: str,
        query:   str,
        top_k:   int = VECTOR_MEMORY_TOP_K,
    ) -> str:
        """
        Search the user's vector memory for past exchanges semantically
        similar to the current query.

        Returns a formatted string block for injection into the system prompt,
        or an empty string if nothing relevant was found.
        """
        # Guard: reject guests and blank ids — no cross-user reads allowed
        self._validate_user_id(user_id)
        collection = self._get_collection(user_id)

        # Ownership check: the collection's metadata must match this user_id.
        # This prevents a logic error elsewhere in the app from accidentally
        # returning one user's memories to another.
        owner = (collection.metadata or {}).get("owner_user_id", "")
        if owner and owner != user_id:
            print(
                f"[VectorMemoryStore.retrieve] SECURITY: user_id={user_id!r} tried "
                f"to read collection owned by {owner!r}. Blocked."
            )
            return ""

        # If the collection is empty, skip
        if collection.count() == 0:
            return ""

        query_embedding = self._model.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "distances"],
        )

        docs      = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not docs:
            return ""

        # Filter out low-relevance results (cosine distance > 1.2 ≈ low similarity)
        relevant = [
            (doc, dist)
            for doc, dist in zip(docs, distances)
            if dist < 1.2
        ]

        if not relevant:
            return ""

        lines = ["[Long-term Memory (past conversations)]"]
        for i, (doc, dist) in enumerate(relevant, 1):
            # Truncate long exchanges for context window efficiency
            truncated = doc[:400] + "…" if len(doc) > 400 else doc
            lines.append(f"  {i}. {truncated}")

        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self, user_id: str) -> Dict[str, int]:
        """Return the number of stored memories for this user only."""
        try:
            self._validate_user_id(user_id)
            collection = self._get_collection(user_id)
            return {"stored_memories": collection.count()}
        except (ValueError, Exception):
            return {"stored_memories": 0}
