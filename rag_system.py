"""
rag_system.py
=============
Core RAG engine with multi-layer memory integration.
"""

from __future__ import annotations

import os
import re
import uuid
import numpy as np
import psycopg
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, MessagesState, StateGraph
from psycopg_pool import ConnectionPool

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import (
    EMBEDDING_MODEL, GROQ_API_KEY, GROQ_MODEL,
    CHROMA_PATH, COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
    HYBRID_TOP_K, ALPHA, RERANK_TOP_K,
    DATABASE_URL,
    MAX_QUERY_LENGTH, RELEVANCE_THRESHOLD,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
)
from memory_manager import MemoryManager
from vector_memory import VectorMemoryStore

# ── Optional Neo4j driver ─────────────────────────────────────────────────────
try:
    from neo4j import GraphDatabase as _Neo4jGraphDatabase
    _NEO4J_DRIVER_AVAILABLE = True
except ImportError:
    _NEO4J_DRIVER_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants / Regexes
# ─────────────────────────────────────────────────────────────────────────────

FALLBACK_RESPONSE = (
    "I'm afraid I don't have sufficient information in the document "
    "to answer that with confidence. Might you rephrase, or perhaps "
    "upload a relevant document first?"
)

_PAGE_QUERY_RE = re.compile(r'\bpage\s+(\d+)\b', re.IGNORECASE)
_CONVERSATIONAL_RE = re.compile(
    r'\b(which|what)\b.{0,30}\b(question|ask|asked)\b.{0,30}\b(last|previous|before|earlier|first|my|I)\b',
    re.IGNORECASE,
)
_LATEST_DOC_RE = re.compile(
    r'\b(latest|last|recent|newest|new|second|current|just)\b.{0,30}'
    r'\b(document|doc|pdf|file|upload|uploaded|paper)\b',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid search helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def hybrid_search(
    query: str,
    st_model: SentenceTransformer,
    docs_embed: np.ndarray,
    bm25_index: BM25Okapi,
    top_k: int = HYBRID_TOP_K,
    alpha: float = ALPHA,
) -> list[int]:
    query_embed   = st_model.encode(query, normalize_embeddings=True)
    dense_scores  = np.dot(docs_embed, query_embed)
    tokens        = query.lower().split()
    sparse_scores = np.array(bm25_index.get_scores(tokens))
    combined      = alpha * _normalize(dense_scores) + (1 - alpha) * _normalize(sparse_scores)
    return np.argsort(combined)[::-1][:top_k].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# RAGEngine
# ─────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Central engine.  Call .chat() for a full conversation turn.

    Parameters
    ----------
    user_id  : str | None  - authenticated username; None → guest
    is_guest : bool        - True for unauthenticated sessions
    """

    def __init__(self):
        # ── Embedding & retrieval ─────────────────────────────────────────────
        self.st_model   = SentenceTransformer(EMBEDDING_MODEL)
        self.embeddings = SentenceTransformerEmbeddings(EMBEDDING_MODEL)

        # ── LLM ───────────────────────────────────────────────────────────────
        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("Groq API key not found.")
        self.llm = ChatGroq(api_key=api_key, model_name=GROQ_MODEL, temperature=0.3)

        # ── ChromaDB ──────────────────────────────────────────────────────────
        # Ensure the storage directory exists before ChromaDB tries to open it.
        # Without this, PersistentClient raises "Could not connect to tenant
        # default_tenant" on a fresh checkout where data/chroma doesn't exist yet.
        os.makedirs(CHROMA_PATH, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # in-memory cache
        self._chunks:     list[str]              = []
        self._docs_embed: np.ndarray             = np.empty((0,))
        self._bm25_index: BM25Okapi | None       = None
        self._metadata:   list[Dict[str, Any]]   = []

        # Track uploaded documents in order: [(doc_name, upload_order), …]
        self._uploaded_docs: list[Tuple[str, int]] = []

        # ── Reload any previously-indexed documents from ChromaDB ─────────────
        self._load_existing_collection()

        # ── LangGraph + Postgres checkpointer ────────────────────────────────
        db_uri = os.getenv("DATABASE_URL", DATABASE_URL)
        try:
            with psycopg.connect(db_uri, autocommit=True) as setup_conn:
                PostgresSaver(setup_conn).setup()
        except Exception as exc:
            print(f"PostgresSaver.setup() notice (safe to ignore): {exc}")

        self.pool   = ConnectionPool(conninfo=db_uri, max_size=20, timeout=30)
        self.pg_mem = PostgresSaver(self.pool)

        # ── Vector Store for long-term per-user memory ─────────────────────
        self.vector_store = VectorMemoryStore(
            chroma_client=self.chroma_client,
            st_model=self.st_model,
        )

        # ── Neo4j driver (for persistent Knowledge Graph) ──────────────────
        self._neo4j_driver = None
        if _NEO4J_DRIVER_AVAILABLE:
            try:
                self._neo4j_driver = _Neo4jGraphDatabase.driver(
                    NEO4J_URI,
                    auth=(NEO4J_USER, NEO4J_PASSWORD),
                )
                self._neo4j_driver.verify_connectivity()
                print("[RAGEngine] Neo4j connected ✓")
            except Exception as exc:
                print(f"[RAGEngine] Neo4j unavailable, falling back to in-memory KG: {exc}")
                self._neo4j_driver = None
        else:
            print("[RAGEngine] neo4j driver not installed, using in-memory KG")

        # ── Multi-layer memory manager ────────────────────────────────────────
        self.memory_mgr = MemoryManager(
            llm=self.llm,
            vector_store=self.vector_store,
            neo4j_driver=self._neo4j_driver,
        )

        # ── Build the LangGraph workflow ──────────────────────────────────────
        self.workflow = self._build_workflow()
        self.chatbot  = self.workflow.compile(checkpointer=self.pg_mem)

    # ── PDF ingestion ─────────────────────────────────────────────────────────

    def process_pdf(
        self,
        pdf_path:   str,
        doc_name:   str = "",
        session_id: str = "global",
    ) -> None:
        """
        Index a PDF.  *doc_name* is the user-facing filename.

        Parameters
        ----------
        session_id : The owner of these chunks.
                     Use the authenticated ``user_id`` for logged-in users,
                     or the ``guest_thread_id`` (e.g. ``"guest_abc123"``) for
                     guests.  Every chunk is tagged with this value so it can
                     be retrieved and purged in isolation.
        """
        loader   = PyPDFLoader(pdf_path)
        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks: list[Document] = splitter.split_documents(pages)

        # Assign a sequential upload order so we can find "the latest doc"
        upload_order = (max(o for _, o in self._uploaded_docs) + 1) if self._uploaded_docs else 1
        if not doc_name:
            doc_name = os.path.basename(pdf_path)
        self._uploaded_docs.append((doc_name, upload_order))

        collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
        for idx, chunk in enumerate(chunks):
            # Enrich metadata with document identity AND session ownership
            meta = dict(chunk.metadata)
            meta["doc_name"]     = doc_name
            meta["upload_order"] = upload_order
            meta["session_id"]   = session_id   # ← ownership tag
            collection.add(
                documents=[chunk.page_content],
                metadatas=[meta],
                ids=[f"chunk_{len(self._chunks) + idx}"],
            )

        self._chunks.extend([c.page_content for c in chunks])
        self._metadata.extend(
            [
                {
                    **c.metadata,
                    "doc_name":     doc_name,
                    "upload_order": upload_order,
                    "session_id":   session_id,
                }
                for c in chunks
            ]
        )
        self._rebuild_hybrid_indices()
        print(
            f"[RAGEngine] Indexed '{doc_name}' (upload #{upload_order}, "
            f"session={session_id!r}) — {len(chunks)} chunks"
        )

    # ── Collection management ─────────────────────────────────────────────────

    def clear_all_data(self) -> bool:
        try:
            self.chroma_client.delete_collection(COLLECTION_NAME)
            self._chunks        = []
            self._docs_embed    = np.empty((0,))
            self._bm25_index    = None
            self._metadata      = []
            self._uploaded_docs  = []
            return True
        except Exception as exc:
            print(f"Error clearing data: {exc}")
            return False

    def clear_guest_data(self, session_id: str) -> int:
        """
        Purge all ChromaDB chunks that belong to a specific guest session,
        then rebuild the in-memory hybrid indices without those chunks.

        Called on guest logout / new guest session so a subsequent visitor
        never inherits a previous guest's indexed documents.

        Returns the number of chunks removed.
        """
        if not session_id:
            return 0
        try:
            collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
            # ChromaDB where-filter: fetch IDs for chunks owned by this session
            result = collection.get(
                where={"session_id": {"$eq": session_id}},
                include=[],   # IDs only — no text needed
            )
            ids_to_delete = result.get("ids", [])
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                print(
                    f"[RAGEngine] Purged {len(ids_to_delete)} guest chunks "
                    f"for session {session_id!r}"
                )
        except Exception as exc:
            print(f"[RAGEngine.clear_guest_data] ChromaDB delete failed: {exc}")
            ids_to_delete = []

        # Rebuild in-memory state by dropping guest chunks from the local lists
        if ids_to_delete:
            # _metadata mirrors ChromaDB; filter out guest rows by session_id
            surviving_pairs = [
                (chunk, meta)
                for chunk, meta in zip(self._chunks, self._metadata)
                if meta.get("session_id") != session_id
            ]
            if surviving_pairs:
                self._chunks, self._metadata = map(list, zip(*surviving_pairs))
            else:
                self._chunks, self._metadata = [], []

            # Rebuild _uploaded_docs from surviving metadata
            seen: Dict[str, int] = {}
            for m in self._metadata:
                name  = m.get("doc_name", "unknown")
                order = m.get("upload_order", 0)
                if name not in seen or order > seen[name]:
                    seen[name] = order
            self._uploaded_docs = sorted(seen.items(), key=lambda x: x[1])

            self._rebuild_hybrid_indices()

        return len(ids_to_delete)

    def load_session_data(self, session_id: str) -> None:
        """
        Reload only the chunks that belong to *session_id* from ChromaDB
        into the in-memory hybrid indices.

        Call this after login (pass the user_id) so the engine serves
        documents the user previously indexed — without exposing any other
        user's or guest's chunks.
        """
        self._chunks        = []
        self._docs_embed    = np.empty((0,))
        self._bm25_index    = None
        self._metadata      = []
        self._uploaded_docs = []

        try:
            collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
            count = collection.count()
            if count == 0:
                return

            result = collection.get(
                where={"session_id": {"$eq": session_id}},
                include=["documents", "metadatas"],
            )

            docs      = result.get("documents", []) or []
            metadatas = result.get("metadatas", []) or []

            if not docs:
                return

            self._chunks   = list(docs)
            self._metadata = list(metadatas)

            # Reconstruct _uploaded_docs from metadata
            seen: Dict[str, int] = {}
            for m in metadatas:
                name  = m.get("doc_name", "unknown")
                order = m.get("upload_order", 0)
                if name not in seen or order > seen[name]:
                    seen[name] = order
            self._uploaded_docs = sorted(seen.items(), key=lambda x: x[1])

            self._rebuild_hybrid_indices()
            print(
                f"[RAGEngine] Loaded {len(self._chunks)} chunks for "
                f"session={session_id!r} "
                f"({len(self._uploaded_docs)} document(s))"
            )

        except Exception as exc:
            print(f"[RAGEngine.load_session_data] {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_existing_collection(self) -> None:
        """
        Reload documents from ChromaDB into the in-memory cache.

        Only called at engine startup.  Because chunks are now tagged with
        ``session_id``, this method loads ALL persisted chunks so that logged-in
        users can subsequently call ``load_session_data(user_id)`` to filter
        to their own.  Guest sessions start with an EMPTY cache — ``process_pdf``
        will populate their view, and ``clear_guest_data`` will wipe it on logout.

        Legacy chunks that pre-date the ``session_id`` field are tagged with
        ``"global"`` so they remain accessible until explicitly cleared.
        """
        try:
            collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
            count = collection.count()
            if count == 0:
                return

            result = collection.get(
                include=["documents", "metadatas"],
                limit=count,
            )

            docs      = result.get("documents", []) or []
            metadatas = result.get("metadatas", []) or []

            if not docs:
                return

            # Back-fill session_id = "global" for any legacy chunks that lack it
            updated_ids   = []
            updated_metas = []
            all_ids       = result.get("ids", [])
            for i, (doc_id, meta) in enumerate(zip(all_ids, metadatas)):
                if "session_id" not in meta:
                    meta["session_id"] = "global"
                    updated_ids.append(doc_id)
                    updated_metas.append(meta)

            if updated_ids:
                collection.update(ids=updated_ids, metadatas=updated_metas)
                print(
                    f"[RAGEngine] Back-filled session_id='global' on "
                    f"{len(updated_ids)} legacy chunks"
                )

            # Do NOT load any chunks into the active session at engine startup.
            # Authenticated users call load_session_data(user_id) after login.
            # Guests start empty — they only see what they upload this session.
            print(
                f"[RAGEngine] ChromaDB has {count} persisted chunk(s). "
                "Call load_session_data(session_id) to activate a session's documents."
            )

        except Exception as exc:
            print(f"[RAGEngine] Could not inspect existing collection: {exc}")

    def _rebuild_hybrid_indices(self) -> None:
        if not self._chunks:
            return
        self._docs_embed = self.st_model.encode(
            self._chunks, normalize_embeddings=True, show_progress_bar=False
        )
        self._bm25_index = BM25Okapi([c.lower().split() for c in self._chunks])

    def _apply_input_guardrails(self, query: str) -> Tuple[bool, str]:
        if not query or not query.strip():
            return False, "I beg your pardon — it seems you submitted an empty query."
        if len(query) > MAX_QUERY_LENGTH:
            return False, (
                f"My sincerest apologies, but your query exceeds "
                f"{MAX_QUERY_LENGTH} characters. Might you condense it?"
            )
        return True, ""

    def _is_conversational_query(self, query: str) -> bool:
        return bool(_CONVERSATIONAL_RE.search(query))

    def _wants_latest_doc(self, query: str) -> bool:
        """Return True if the user is asking about the latest/last uploaded document."""
        return bool(_LATEST_DOC_RE.search(query))

    def _get_latest_doc_name(self) -> Optional[str]:
        """Return the name of the most recently uploaded document, if any."""
        if not self._uploaded_docs:
            return None
        return self._uploaded_docs[-1][0]

    def _active_docs_info(self) -> str:
        """Return a summary of currently indexed documents for the system prompt."""
        if not self._uploaded_docs:
            return ""
        names = [name for name, _ in self._uploaded_docs]
        latest = names[-1]
        if len(names) == 1:
            return f"[Currently Loaded Document: \"{latest}\" — {len(self._chunks)} chunks indexed]"
        return (
            f"[Currently Loaded Documents: {', '.join(names)} — "
            f"{len(self._chunks)} chunks total. "
            f"Most recently uploaded: \"{latest}\"]"
        )

    def _extract_page_number(self, query: str) -> Optional[int]:
        m = _PAGE_QUERY_RE.search(query)
        return int(m.group(1)) if m else None

    def _get_hybrid_context(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
        if not self._chunks or self._bm25_index is None:
            return "", 0.0

        top_indices = hybrid_search(query, self.st_model, self._docs_embed, self._bm25_index)

        if metadata_filter:
            filtered = [
                i for i in top_indices
                if i < len(self._metadata)
                and all(self._metadata[i].get(k) == v for k, v in metadata_filter.items())
            ]
            if filtered:
                top_indices = filtered

        if not top_indices:
            return "", 0.0

        query_embed = self.st_model.encode(query, normalize_embeddings=True)
        best_score  = float(np.dot(self._docs_embed[top_indices[0]], query_embed))

        parts = []
        for i in top_indices:
            meta = self._metadata[i] if i < len(self._metadata) else {}
            page = meta.get("page", "?")
            doc  = meta.get("doc_name", "")
            label = f"[{doc} — Page {page}]" if doc else f"[Page {page}]"
            parts.append(f"{label}\n{self._chunks[i]}")

        return "\n\n---\n\n".join(parts), best_score

    # ── LangGraph workflow ────────────────────────────────────────────────────

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(state_schema=MessagesState)

        _BUTLER_BASE = (
            "You are an Elegant Butler from England. Answer all questions with grace, "
            "politeness, and sophistication."
        )

        # Maximum number of old messages to keep from the LangGraph checkpointer.
        # This prevents stale conversation history (about previously loaded docs)
        # from overriding the current document's RAG context.
        _MAX_HISTORY_MSGS = 6

        def call_model(state: MessagesState, config: RunnableConfig) -> Dict[str, Any]:
            # ── Read memory prefix injected by chat() ─────────────────────────
            memory_prefix: str = (
                config.get("configurable", {}).get("memory_prefix", "") or ""
            )

            # ── Extract last user message ─────────────────────────────────────
            last_human = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_human = msg.content
                    break

            context, relevance_score = "", 0.0
            metadata_filter: Optional[Dict[str, Any]] = None

            if last_human:
                page_num = self._extract_page_number(last_human)
                if page_num is not None:
                    metadata_filter = {"page": page_num}

                # If user asks about the latest/last document, filter to it
                if self._wants_latest_doc(last_human):
                    latest = self._get_latest_doc_name()
                    if latest:
                        metadata_filter = metadata_filter or {}
                        metadata_filter["doc_name"] = latest

                # Only treat as conversational if there are NO indexed
                # document chunks — otherwise always try RAG first.
                if not self._chunks and self._is_conversational_query(last_human):
                    system_msg = _BUTLER_BASE
                    if memory_prefix:
                        system_msg = memory_prefix + "\n\n" + system_msg
                    msgs = [SystemMessage(content=system_msg)] + list(state["messages"])
                    return {"messages": [self.llm.invoke(msgs)]}

                context, relevance_score = self._get_hybrid_context(
                    last_human, metadata_filter=metadata_filter
                )

            # ── Active document header ────────────────────────────────────────
            doc_header = self._active_docs_info()

            # ── Build system prompt ───────────────────────────────────────────
            # When document chunks are indexed, ALWAYS include the retrieved
            # context.  The old behavior of dropping context when the
            # relevance score was low caused the LLM to fall back to stale
            # conversation history about previously-loaded documents.
            if context and self._chunks:
                page_instruction = ""
                if metadata_filter and "page" in metadata_filter:
                    page_instruction = (
                        f"The user is asking about PAGE {metadata_filter['page']}. "
                        "Each chunk is labelled [Page N] — trust these labels completely.\n\n"
                    )

                confidence = "HIGH" if relevance_score >= RELEVANCE_THRESHOLD else "MODERATE"

                system_msg = (
                    (doc_header + "\n\n" if doc_header else "")
                    + (memory_prefix + "\n\n" if memory_prefix else "")
                    + _BUTLER_BASE + "\n\n"
                    + page_instruction
                    + f"Relevance confidence: {confidence}.\n"
                    + "IMPORTANT: Answer ONLY from the DOCUMENT CONTEXT provided below. "
                    + "Do NOT use information from previous conversations about "
                    + "other documents. The context below is from the CURRENTLY "
                    + "loaded document(s).\n\n"
                    + f"CONTEXT:\n{context}"
                )
            elif self._chunks:
                # Chunks exist but no matching context was retrieved
                # (e.g. empty query) — still inform about active doc
                system_msg = (
                    (doc_header + "\n\n" if doc_header else "")
                    + (memory_prefix + "\n\n" if memory_prefix else "")
                    + _BUTLER_BASE + "\n\n"
                    + "A document is loaded but the query did not match any "
                    + "specific content. Answer based on general knowledge "
                    + "or ask the user to rephrase."
                )
            else:
                # No documents indexed — answer from memory + persona
                system_msg = (
                    (memory_prefix + "\n\n" if memory_prefix else "")
                    + _BUTLER_BASE
                )

            # ── Trim old messages to prevent stale doc history ────────────────
            # The LangGraph checkpointer replays ALL previous messages in the
            # thread.  If the user discussed a different document earlier,
            # those old exchanges would overwhelm the current RAG context.
            # We keep only the most recent messages.
            all_msgs = list(state["messages"])
            if len(all_msgs) > _MAX_HISTORY_MSGS:
                all_msgs = all_msgs[-_MAX_HISTORY_MSGS:]

            msgs = [SystemMessage(content=system_msg)] + all_msgs
            return {"messages": [self.llm.invoke(msgs)]}

        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        return workflow

    # ── Public chat interface ─────────────────────────────────────────────────

    def chat(
        self,
        query:     str,
        thread_id: str = "default",
        user_id:   Optional[str] = None,
        is_guest:  bool = False,
    ) -> str:
        """
        Main entry point.

        Parameters
        ----------
        query     : user input text
        thread_id : LangGraph checkpointer key (unique per conversation)
        user_id   : authenticated username (None for guests)
        is_guest  : True when user is not authenticated
        """
        config = {"configurable": {"thread_id": thread_id}}

        # ── Guardrails ────────────────────────────────────────────────────────
        valid, err = self._apply_input_guardrails(query)
        if not valid:
            return err

        # ── Memory context (ChatGPT-style 4-layer injection) ──────────────────
        # buffer_msgs  = Layer 4: current session as BaseMessage list
        # system_prefix = Layers 1-3 + KG: session metadata, user memory,
        #                 recent conversation summaries, knowledge graph facts
        buffer_msgs, system_prefix = self.memory_mgr.build_context(
            query     = query,
            user_id   = user_id,
            thread_id = thread_id,
            is_guest  = is_guest,
        )

        # ── Inject buffer history + invoke chatbot ────────────────────────────
        # system_prefix is stored in the config so the workflow can read it
        config["configurable"]["memory_prefix"] = system_prefix
        messages: list = list(buffer_msgs) + [HumanMessage(content=query)]
        output   = self.chatbot.invoke({"messages": messages}, config)
        response = output["messages"][-1].content

        # ── Record exchange in memory layers ──────────────────────────────────
        self.memory_mgr.record_exchange(
            human     = query,
            assistant = response,
            user_id   = user_id,
            thread_id = thread_id,
            is_guest  = is_guest,
        )

        return response

    # ── New chat: create a fresh thread while keeping long-term memory ────────

    def new_chat_thread(
        self,
        user_id:   Optional[str] = None,
        is_guest:  bool = False,
        thread_id: Optional[str] = None,
    ) -> str:
        """
        End the current conversation and start a fresh one.

        ChatGPT behaviour replicated:
          1. Summarise the just-finished conversation → stored in Layer 3
             (Recent Conversations).  Buffer (Layer 4) is cleared.
          2. User Memory (Layer 2) and KG are preserved across new chats.
          3. A brand-new thread_id is issued for LangGraph checkpointing.
        """
        new_id = str(uuid.uuid4())
        if not is_guest and user_id:
            # Summarise + clear current session
            self.memory_mgr.close_and_summarise(
                user_id   = user_id,
                thread_id = thread_id or user_id,
            )
        return new_id

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def memory_diagnostics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return self.memory_mgr.diagnostics(user_id)

    def list_pages(self) -> list[int]:
        pages = set()
        for meta in self._metadata:
            p = meta.get("page")
            if p is not None:
                pages.add(int(p))
        return sorted(pages)