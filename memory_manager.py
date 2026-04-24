"""
memory_manager.py  -  Multi-layer memory architecture
=========================================================

Memory Layers:
  
    GLOBAL MEMORY (shared across ALL users and sessions)            
      • Global Summary Memory  - condensed knowledge from all chats 
      • Global Knowledge Graph - structured entity-relation triples 
  
    PER-CONVERSATION MEMORY                                         
                                                                    
    Guest Users:                                                    
      • Buffer Window Memory (short-term context only)              
                                                                    
    Logged-in Users:                                                
      • Buffer Window Memory   - recent messages (current session)  
      • Summary Memory         - conversation-level summarisation   
      • User Knowledge Graph   - user-specific structured memory    
      • Vector Store Retriever - semantic long-term memory retrieval
  

Context injection order (into every prompt):
  1. Session Metadata        - device / timezone / usage stats  (ephemeral)
  2. User Memory             - explicit persistent facts          (permanent)
  3. Recent Conversations    - ~15 past-chat dated summaries      (pre-computed)
  4. User Knowledge Graph    - user-specific entity triples       (per-user)
  5. Global Summary Memory   - condensed knowledge from all users (shared)
  6. Global Knowledge Graph  - domain entity triples              (shared)
  7. Vector Store Retriever  - semantically similar past exchanges(per-user)
  8. Current Session Messages- full raw transcript of THIS chat   (in-context)

Retrieval fallback order:
  Buffer Window → Global Memory + User Memory → Vector Store Retriever
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

try:
    from memori import Memori
    _MEMORI_AVAILABLE = True
except ImportError:
    _MEMORI_AVAILABLE = False

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    BUFFER_WINDOW_SIZE, SUMMARY_THRESHOLD,
    MEMORI_API_KEY,
)
from global_memory import GlobalSummaryMemory, GlobalKnowledgeGraph

try:
    from neo4j_kg import Neo4jKnowledgeGraph, Neo4jGlobalKnowledgeGraph
    _NEO4J_KG_AVAILABLE = True
except ImportError:
    _NEO4J_KG_AVAILABLE = False
from vector_memory import VectorMemoryStore

# How many past-conversation summaries to inject 
RECENT_CONV_LIMIT = 15

# Max facts stored in User Memory 
MAX_USER_FACTS = 50


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 – Session Metadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionMetadata:
    """
    Ephemeral per-session context injected once at the start of every chat.
    Mirrors what ChatGPT injects: device info, timezone, subscription tier,
    usage frequency, etc.  Disappears when the session ends.
    """
    user_id:           str
    session_id:        str        = field(default_factory=lambda: uuid.uuid4().hex[:8])
    subscription_tier: str        = "free"            # "free" | "pro"
    session_start:     str        = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )
    # populated by the app layer
    total_conversations: int      = 0
    total_messages:      int      = 0
    avg_messages_per_conv: float  = 0.0
    last_active:         str      = ""

    def to_prompt_block(self) -> str:
        lines = [
            "[Session Metadata]",
            f"• User: {self.user_id}",
            f"• Subscription: {self.subscription_tier}",
            f"• Session started: {self.session_start}",
            f"• Total past conversations: {self.total_conversations}",
            f"• Avg messages/conversation: {self.avg_messages_per_conv:.1f}",
        ]
        if self.last_active:
            lines.append(f"• Last active: {self.last_active}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 – User Memory  (explicit persistent facts)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UserFact:
    text:       str
    source:     str   = "auto"    # "auto" | "explicit"
    created_at: str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fact_id:    str   = field(default_factory=lambda: uuid.uuid4().hex[:6])


class UserMemory:
    """
    Permanent fact store — mirrors ChatGPT's explicit memory.

    Facts are stored deliberately:
      • User says "remember that …"   → explicit
      • LLM detects a salient detail  → auto  (confirmed by conversation flow)

    All facts are injected into every prompt, every turn.
    """

    _EXTRACT_PROMPT = (
        "You are a memory extraction assistant.\n"
        "Read the exchange below and decide if there is ONE important, "
        "durable fact about the user worth remembering long-term "
        "(e.g. name, job, goal, preference, project they are building).\n"
        "If yes, output ONLY the fact as a single sentence starting with the user's name "
        "or 'The user'. If no durable fact exists, output exactly: NONE\n\n"
        "Exchange:\nUser: {human}\nAssistant: {assistant}"
    )

    def __init__(self):
        self._facts: List[UserFact] = []

    # Mutations 
    def remember(self, text: str, source: str = "auto") -> UserFact:
        """Add a fact; silently deduplicate near-identical entries."""
        text = text.strip()
        for f in self._facts:
            if f.text.lower() == text.lower():
                return f                          # already known
        fact = UserFact(text=text, source=source)
        self._facts.append(fact)
        if len(self._facts) > MAX_USER_FACTS:
            # drop oldest auto facts first
            auto = [f for f in self._facts if f.source == "auto"]
            if auto:
                self._facts.remove(auto[0])
        return fact

    def forget(self, fact_id: str) -> bool:
        before = len(self._facts)
        self._facts = [f for f in self._facts if f.fact_id != fact_id]
        return len(self._facts) < before

    def forget_by_text(self, text: str) -> bool:
        before = len(self._facts)
        self._facts = [f for f in self._facts if text.lower() not in f.text.lower()]
        return len(self._facts) < before

    def all_facts(self) -> List[UserFact]:
        return list(self._facts)


    # Auto-extraction from exchange 
    def try_extract(self, human: str, assistant: str, llm: ChatGroq) -> Optional[str]:
        """Ask the LLM if this exchange contains a durable fact to store."""
        prompt = self._EXTRACT_PROMPT.format(human=human[:800], assistant=assistant[:800])
        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if raw.upper() == "NONE" or not raw:
                return None
            # Sanity: must be a complete sentence, not a question
            if "?" not in raw and len(raw) > 10:
                self.remember(raw, source="auto")
                return raw
        except Exception as exc:
            print(f"[UserMemory.try_extract] {exc}")
        return None


    # Prompt block 
    def to_prompt_block(self) -> str:
        if not self._facts:
            return ""
        lines = ["[User Memory]"]
        for f in self._facts:
            tag = "(explicit)" if f.source == "explicit" else ""
            lines.append(f"• {f.text} {tag}".strip())
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._facts)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 – Recent Conversations Summary  (pre-computed, NOT RAG)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationSummary:
    """A brief summary of one past conversation — mirrors ChatGPT's format."""
    date:       str          # "Apr 22, 2026"
    title:      str          # "Building a load balancer in Go"
    bullets:    List[str]    # ["asked about connection pooling", …]
    thread_id:  str  = ""
    created_at: str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RecentConversations:
    """
    Keeps a rolling list of the last RECENT_CONV_LIMIT conversation summaries.

    On conversation close (new chat clicked) → generate_summary() is called.
    The summary is then prepended to the list for future sessions.

    ChatGPT only summarises what the USER said, not the assistant.
    We follow the same convention.
    """

    _SUMMARISE_PROMPT = (
        "You are a conversation archivist.\n"
        "Below is a conversation. Produce:\n"
        "  1. A SHORT title (≤ 6 words) describing the main topic.\n"
        "  2. Up to 3 bullet points summarising ONLY what the USER wanted "
        "or discussed (ignore assistant responses).\n\n"
        "Return ONLY valid JSON in this exact shape:\n"
        '{{"title": "...", "bullets": ["...", "..."]}}\n\n'
        "Conversation:\n{history}"
    )

    def __init__(self):
        self._summaries: Deque[ConversationSummary] = deque(maxlen=RECENT_CONV_LIMIT)

    def generate_and_store(
        self,
        pairs:     List[Tuple[str, str]],   # [(human, assistant), …]
        thread_id: str,
        llm:       ChatGroq,
    ) -> Optional[ConversationSummary]:
        """Summarise a finished conversation and add it to the store."""
        if not pairs:
            return None

        history = "\n".join(f"User: {h}\nAssistant: {a}" for h, a in pairs)
        try:
            prompt  = self._SUMMARISE_PROMPT.format(history=history[:3000])
            raw = llm.invoke([HumanMessage(content=prompt)]).content
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            # Try to extract a JSON object from the response
            # Sometimes the LLM wraps it in extra text
            json_match = re.search(r'\{[^{}]*"title"[^{}]*\}', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            data = json.loads(raw)
            title = data.get("title", "Conversation") if isinstance(data, dict) else "Conversation"
            bullets = data.get("bullets", []) if isinstance(data, dict) else []
            summary = ConversationSummary(
                date      = datetime.now(timezone.utc).strftime("%b %d, %Y"),
                title     = str(title)[:60],
                bullets   = [str(b)[:120] for b in bullets[:3]],
                thread_id = thread_id,
            )
            self._summaries.appendleft(summary)   # newest first
            return summary
        except Exception as exc:
            print(f"[RecentConversations.generate_and_store] {exc}")
            # Fallback: create a basic summary even if LLM parsing fails
            fallback = ConversationSummary(
                date      = datetime.now(timezone.utc).strftime("%b %d, %Y"),
                title     = f"Conversation ({len(pairs)} turns)",
                bullets   = [pairs[0][0][:120]] if pairs else [],
                thread_id = thread_id,
            )
            self._summaries.appendleft(fallback)
            return fallback

    def to_prompt_block(self) -> str:
        if not self._summaries:
            return ""
        lines = ["[Recent Conversations]"]
        for i, s in enumerate(self._summaries, 1):
            lines.append(f"{i}. {s.date}: \"{s.title}\"")
            for b in s.bullets:
                lines.append(f"   - {b}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._summaries)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 4 – Current Session Messages  (raw in-context transcript)
# ─────────────────────────────────────────────────────────────────────────────

class CurrentSession:
    """
    Plain buffer of this conversation's messages.
    No summarisation.  Trimmed from the oldest end if context runs out.
    """

    def __init__(self, window_size: int = BUFFER_WINDOW_SIZE):
        self.window_size = window_size
        self._pairs: List[Tuple[str, str]] = []   # (human, assistant)

    def add(self, human: str, assistant: str) -> None:
        self._pairs.append((human, assistant))
        if len(self._pairs) > self.window_size:
            self._pairs.pop(0)                   # drop oldest

    def get_messages(self) -> List[BaseMessage]:
        msgs: List[BaseMessage] = []
        for h, a in self._pairs:
            msgs.append(HumanMessage(content=h))
            msgs.append(AIMessage(content=a))
        return msgs

    def get_pairs(self) -> List[Tuple[str, str]]:
        return list(self._pairs)

    def clear(self) -> List[Tuple[str, str]]:
        """Clear and return pairs (so caller can summarise them)."""
        pairs = list(self._pairs)
        self._pairs = []
        return pairs

    def __len__(self) -> int:
        return len(self._pairs)


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Graph  (per-user structured entity store)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KGEdge:
    source:   str
    relation: str
    target:   str


class KnowledgeGraph:
    """Lightweight in-memory KG for structured entity/relation facts."""

    _EXTRACT_PROMPT = (
        "Extract knowledge graph triples from the text below.\n"
        "Return ONLY a JSON array of objects with keys: "
        '"subject", "relation", "object".\n'
        "Keep each value ≤ 5 words. Output NOTHING else.\n\n"
        "TEXT:\n{text}"
    )

    def __init__(self):
        self.nodes: Dict[str, str] = {}     # entity → type
        self.edges: List[KGEdge]   = []

    def add(self, subject: str, relation: str, obj: str) -> None:
        s, o = subject.lower().strip(), obj.lower().strip()
        self.nodes.setdefault(s, "Concept")
        self.nodes.setdefault(o, "Concept")
        if not any(e.source == s and e.relation == relation and e.target == o
                   for e in self.edges):
            self.edges.append(KGEdge(s, relation, o))

    def extract_and_store(self, text: str, llm: ChatGroq,
                          max_triples: int = 8) -> None:
        prompt = self._EXTRACT_PROMPT.format(text=text[:1500])
        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            for t in json.loads(raw)[:max_triples]:
                s = str(t.get("subject", "")).strip()
                r = str(t.get("relation", "")).strip()
                o = str(t.get("object", "")).strip()
                if s and r and o:
                    self.add(s, r, o)
        except Exception:
            pass

    def query(self, tokens: List[str]) -> str:
        hits = []
        for e in self.edges:
            if any(tok in e.source or tok in e.target for tok in tokens):
                hits.append(f"{e.source} {e.relation} {e.target}")
        if not hits:
            return ""
        unique = list(dict.fromkeys(hits))[:15]
        return "[Knowledge Graph]\n" + "\n".join(f"• {h}" for h in unique)

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager  –  unified façade
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Single entry-point.  Per-user state is keyed by user_id.
    Guest users get only Buffer Window Memory (current session messages).
    Logged-in users get all layers including Global Memory and Vector Store.

    Memory layer hierarchy:
      ┌─ Global Memory (always available) ───────────────────┐
      │  • Global Summary Memory (shared condensed knowledge)│
      │  • Global Knowledge Graph (shared entity triples)    │
      ├─ Per-User Memory (logged-in only) ───────────────────┤
      │  • User Memory (persistent facts)                    │
      │  • Recent Conversations (pre-computed summaries)     │
      │  • User Knowledge Graph (user-specific triples)      │
      │  • Vector Store Retriever (semantic past exchanges)  │
      ├─ Session Memory ─────────────────────────────────────┤
      │  • Buffer Window (current conversation messages)     │
      └──────────────────────────────────────────────────────┘

    Retrieval priority:
      Buffer Window → Global Memory → Vector Store Retriever
    """

    def __init__(
        self,
        llm:            ChatGroq,
        vector_store:   Optional[VectorMemoryStore] = None,
        neo4j_driver    = None,
    ):
        self.llm = llm
        self._neo4j_driver = neo4j_driver

        # ── GLOBAL MEMORY (shared across all users) ──────────────────────────
        self._global_summary = GlobalSummaryMemory()
        if neo4j_driver and _NEO4J_KG_AVAILABLE:
            self._global_kg = Neo4jGlobalKnowledgeGraph(neo4j_driver)
            print("[MemoryManager] Global KG → Neo4j")
        else:
            self._global_kg = GlobalKnowledgeGraph()
            print("[MemoryManager] Global KG → in-memory")

        # ── VECTOR STORE (per-user semantic long-term memory) ─────────────────
        self._vector_store = vector_store   # None if not configured

        # ── Per-user stores ──────────────────────────────────────────────────
        self._user_memory:       Dict[str, UserMemory]          = {}
        self._recent_convs:      Dict[str, RecentConversations] = {}
        self._kg:                Dict[str, KnowledgeGraph]      = {}
        self._session_meta:      Dict[str, SessionMetadata]     = {}
        self._current_sessions:  Dict[str, CurrentSession]      = {}

        # guest: keyed by thread_id
        self._guest_sessions:    Dict[str, CurrentSession]      = {}

        # usage counters (feeds session metadata)
        self._conv_counts:   Dict[str, int]   = {}
        self._msg_counts:    Dict[str, int]   = {}

        # Track which user_ids have been fully bootstrapped this runtime.
        # A new user who has never logged in before gets empty stores on first
        # access (lazy-init already handles that), but we also explicitly record
        # them here so diagnostics and cross-user access checks can reason
        # about "known users vs unknown intruders".
        self._registered_users: set = set()

    # ── User bootstrap ────────────────────────────────────────────────────────

    def ensure_user_initialized(self, user_id: str) -> None:
        """
        Idempotently bootstrap all per-user memory stores for *user_id*.

        Safe to call on every login — repeated calls are no-ops once the
        user is in _registered_users.  Creates isolated, empty stores the
        first time a new user is seen; never touches any other user's data.
        """
        if not user_id or user_id.startswith("guest_"):
            return  # guests use _guest_sessions, not per-user stores
        if user_id in self._registered_users:
            return  # already set up

        # Touch each store to create empty defaults (lazy-init pattern)
        _ = self._umem(user_id)
        _ = self._rconv(user_id)
        _ = self._get_kg(user_id)
        _ = self._smeta(user_id)
        _ = self._csess(user_id)

        self._registered_users.add(user_id)
        print(f"[MemoryManager] New user bootstrapped: {user_id!r} — "
              f"isolated memory stores created.")

    def _assert_user_owns_session(self, user_id: str, context: str = "") -> None:
        """
        Raise RuntimeError if user_id is not a registered, non-guest user.

        Used as a runtime guard inside methods that must never be called
        for guests or with an unknown/tampered user_id.
        """
        if not user_id or user_id.startswith("guest_"):
            raise RuntimeError(
                f"[MemoryManager{' ' + context if context else ''}] "
                f"Illegal user_id {user_id!r} — guests must not access per-user memory."
            )

    # ── Lazy init ─────────────────────────────────────────────────────────────

    def _umem(self, uid: str)  -> UserMemory:
        return self._user_memory.setdefault(uid, UserMemory())

    def _rconv(self, uid: str) -> RecentConversations:
        return self._recent_convs.setdefault(uid, RecentConversations())

    def _get_kg(self, uid: str):
        if uid not in self._kg:
            if self._neo4j_driver and _NEO4J_KG_AVAILABLE:
                self._kg[uid] = Neo4jKnowledgeGraph(self._neo4j_driver, uid)
            else:
                self._kg[uid] = KnowledgeGraph()
        return self._kg[uid]

    def _smeta(self, uid: str) -> SessionMetadata:
        if uid not in self._session_meta:
            self._session_meta[uid] = SessionMetadata(user_id=uid)
        return self._session_meta[uid]

    def _csess(self, uid: str) -> CurrentSession:
        return self._current_sessions.setdefault(uid, CurrentSession())

    def _gsess(self, tid: str) -> CurrentSession:
        return self._guest_sessions.setdefault(tid, CurrentSession())

    # ── Public: build context ─────────────────────────────────────────────────

    def build_context(
        self,
        query:     str,
        user_id:   Optional[str],
        thread_id: str,
        is_guest:  bool,
    ) -> Tuple[List[BaseMessage], str]:
        """
        Returns
        -------
        buffer_msgs   : Current session as BaseMessage list (Buffer Window)
        system_prefix : All other memory layers as a single string to prepend
                        to the system prompt.

        For guests: only buffer_msgs is populated (Buffer Window Memory only).
        For logged-in: all layers are queried and concatenated.
        """
        if is_guest:
            # ── GUEST: Buffer Window Memory ONLY ─────────────────────────────
            return self._gsess(thread_id).get_messages(), ""

        assert user_id is not None
        # Ensure this user has isolated stores — safe no-op for returning users
        self.ensure_user_initialized(user_id)
        self._assert_user_owns_session(user_id, "build_context")
        prefix_parts: List[str] = []

        # ── Layer 1 – Session Metadata ────────────────────────────────────────
        meta = self._smeta(user_id)
        n_conv = self._conv_counts.get(user_id, 0)
        n_msg  = self._msg_counts.get(user_id, 0)
        meta.total_conversations    = n_conv
        meta.total_messages         = n_msg
        meta.avg_messages_per_conv  = (n_msg / n_conv) if n_conv else 0.0
        prefix_parts.append(meta.to_prompt_block())

        # ── Layer 2 – User Memory (persistent facts) ─────────────────────────
        um_block = self._umem(user_id).to_prompt_block()
        if um_block:
            prefix_parts.append(um_block)

        # ── Layer 3 – Recent Conversations Summary ───────────────────────────
        rc_block = self._rconv(user_id).to_prompt_block()
        if rc_block:
            prefix_parts.append(rc_block)

        # ── Layer 4 – User Knowledge Graph ───────────────────────────────────
        tokens = [w for w in query.lower().split() if len(w) > 3]
        kg_block = self._get_kg(user_id).query(tokens)
        if kg_block:
            prefix_parts.append(kg_block)

        # ── Layer 5 – Global Summary Memory (shared) ─────────────────────────
        global_summary_block = self._global_summary.to_prompt_block()
        if global_summary_block:
            prefix_parts.append(global_summary_block)

        # ── Layer 6 – Global Knowledge Graph (shared) ────────────────────────
        global_kg_block = self._global_kg.query(tokens)
        if global_kg_block:
            prefix_parts.append(global_kg_block)

        # ── Layer 7 – Vector Store Retriever (semantic past exchanges) ───────
        if self._vector_store:
            try:
                vector_block = self._vector_store.retrieve(user_id, query)
                if vector_block:
                    prefix_parts.append(vector_block)
            except ValueError as exc:
                # _validate_user_id raised — should never happen here but log it
                print(f"[MemoryManager.build_context] Vector store access denied: {exc}")

        # ── Layer 8 – Current Session Messages (Buffer Window) ───────────────
        # Returned as BaseMessage list, not part of system prefix
        buf_msgs = self._csess(user_id).get_messages()

        return buf_msgs, "\n\n".join(prefix_parts)

    # ── Public: record exchange ───────────────────────────────────────────────

    def record_exchange(
        self,
        human:     str,
        assistant: str,
        user_id:   Optional[str],
        thread_id: str,
        is_guest:  bool,
    ) -> None:
        """Called after every LLM response to update all memory layers."""
        if is_guest:
            self._gsess(thread_id).add(human, assistant)
            return

        assert user_id is not None
        # Ensure isolated stores exist for this user; guard against spoofed ids
        self.ensure_user_initialized(user_id)
        self._assert_user_owns_session(user_id, "record_exchange")

        # ── Buffer Window – add to current session ───────────────────────────
        self._csess(user_id).add(human, assistant)

        # ── User Memory – try to auto-extract a durable user fact ────────────
        self._umem(user_id).try_extract(human, assistant, self.llm)

        # ── User KG – extract triples ────────────────────────────────────────
        exchange_text = f"User: {human}\nAssistant: {assistant}"
        self._get_kg(user_id).extract_and_store(exchange_text, self.llm)

        # ── Global Summary Memory – extract general knowledge facts ──────────
        self._global_summary.try_extract_from_exchange(
            human, assistant, self.llm, source_user=user_id
        )

        # ── Global Knowledge Graph – extract domain triples ──────────────────
        self._global_kg.extract_and_store(exchange_text, self.llm)

        # ── Usage counters ───────────────────────────────────────────────────
        self._msg_counts[user_id] = self._msg_counts.get(user_id, 0) + 1

    # ── Public: explicit remember / forget (user-driven) ─────────────────────

    def is_new_user(self, user_id: str) -> bool:
        """Return True if this user_id has never been bootstrapped before."""
        return user_id not in self._registered_users

    def remember(self, text: str, user_id: str) -> str:
        """Explicitly store a fact for *this* user only. Returns the fact_id."""
        self.ensure_user_initialized(user_id)
        self._assert_user_owns_session(user_id, "remember")
        fact = self._umem(user_id).remember(text, source="explicit")
        return fact.fact_id

    def forget(self, text_or_id: str, user_id: str) -> bool:
        """Forget a fact by id or partial text match for *this* user only."""
        self.ensure_user_initialized(user_id)
        self._assert_user_owns_session(user_id, "forget")
        um = self._umem(user_id)
        return um.forget(text_or_id) or um.forget_by_text(text_or_id)

    def list_memories(self, user_id: str) -> List[Dict[str, str]]:
        """List memories belonging to *this* user only."""
        self.ensure_user_initialized(user_id)
        self._assert_user_owns_session(user_id, "list_memories")
        return [
            {"id": f.fact_id, "text": f.text, "source": f.source}
            for f in self._umem(user_id).all_facts()
        ]

    # ── Public: new chat ──────────────────────────────────────────────────────

    def close_and_summarise(
        self,
        user_id:   str,
        thread_id: str,
    ) -> Optional[ConversationSummary]:
        """
        Called when user clicks "New Chat":
          1. Grab the current session's pairs.
          2. Generate + store a conversation summary (Layer 3).
          3. Feed the summary into Global Summary Memory (Layer 5).
          4. Store conversation pairs in Vector Store (Layer 7).
          5. Clear the current session buffer (Buffer Window).
          6. Increment conversation counter.

        Each layer is wrapped in try/except so a failure in one layer
        (e.g. LLM rate limit) doesn't block the others.
        """
        sess  = self._csess(user_id)
        # Ensure isolated stores exist — guards against logic errors after logout/login cycles
        self.ensure_user_initialized(user_id)
        pairs = sess.clear()                          # clears Buffer Window

        summary = None
        if pairs:
            # ── Layer 3: Generate conversation summary ───────────────────────
            try:
                summary = self._rconv(user_id).generate_and_store(
                    pairs, thread_id, self.llm
                )
            except Exception as exc:
                print(f"[MemoryManager.close_and_summarise] Layer 3 (summary) failed: {exc}")

            self._conv_counts[user_id] = self._conv_counts.get(user_id, 0) + 1

            # ── Layer 5: Feed summary into Global Summary Memory ─────────────
            if summary:
                try:
                    self._global_summary.try_extract_from_summary(
                        title=summary.title,
                        bullets=summary.bullets,
                        llm=self.llm,
                        source_user=user_id,
                    )
                except Exception as exc:
                    print(f"[MemoryManager.close_and_summarise] Layer 5 (global summary) failed: {exc}")

            # ── Layer 7: Store pairs in Vector Store for long-term recall ────
            if self._vector_store:
                try:
                    stored = self._vector_store.store_conversation(
                        user_id=user_id,
                        thread_id=thread_id,
                        pairs=pairs,
                    )
                    print(f"[MemoryManager] Stored {stored} exchanges in Vector Store for {user_id!r}")
                except ValueError as exc:
                    print(f"[MemoryManager.close_and_summarise] Vector store rejected user: {exc}")
                except Exception as exc:
                    print(f"[MemoryManager.close_and_summarise] Layer 7 (vector store) failed: {exc}")


        # ── Clear the live summary cache for this user ────────────────────────
        cache_key = f"{user_id}_live"
        if hasattr(self, '_live_summary_cache') and cache_key in self._live_summary_cache:
            del self._live_summary_cache[cache_key]
        if hasattr(self, '_live_summary_count') and cache_key in self._live_summary_count:
            del self._live_summary_count[cache_key]

        return summary

    def clear_guest_session(self, thread_id: str) -> None:
        if thread_id in self._guest_sessions:
            self._guest_sessions[thread_id].clear()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            # Global memory stats (always present)
            "global_summary_facts":  len(self._global_summary),
            "global_kg_stats":       self._global_kg.stats(),
        }
        if not user_id:
            return base

        # Per-user stats
        archived_convs = len(self._rconv(user_id))
        # Count the current live session as an active conversation if it has turns
        has_active_session = len(self._csess(user_id)) > 0
        base["user_memory_facts"]   = len(self._umem(user_id))
        base["recent_convs_stored"] = archived_convs + (1 if has_active_session else 0)
        base["has_active_session"]  = has_active_session
        base["archived_convs"]      = archived_convs
        base["kg_stats"]            = self._get_kg(user_id).stats()
        base["session_turns"]       = len(self._csess(user_id))
        base["total_conversations"] = self._conv_counts.get(user_id, 0)
        base["total_messages"]      = self._msg_counts.get(user_id, 0)

        # Vector Store stats
        if self._vector_store:
            base["vector_store_stats"] = self._vector_store.stats(user_id)
        else:
            base["vector_store_stats"] = {"stored_memories": 0}

        return base

    # ── Public accessors for UI previews ──────────────────────────────────────

    def get_buffer_window_pairs(self, user_id: str) -> List[Tuple[str, str]]:
        """Return (human, assistant) pairs in the current session buffer."""
        return self._csess(user_id).get_pairs()

    def get_recent_conversation_summaries(self, user_id: str) -> List[Dict[str, Any]]:
        """Return list of recent conversation summaries for the UI,
        including a live summary for the current active session."""
        results = []

        # Include a live summary for the current active session
        sess = self._csess(user_id)
        if len(sess) > 0:
            live_summary = self._generate_live_summary(user_id)
            if live_summary:
                results.append(live_summary)

        # Add archived conversation summaries
        rconv = self._rconv(user_id)
        for s in rconv._summaries:
            results.append({
                "date": s.date,
                "title": s.title,
                "bullets": s.bullets,
                "thread_id": s.thread_id,
            })

        return results

    def _generate_live_summary(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Generate a live preview summary for the current active session
        without clearing the buffer. Uses a cache to avoid repeated LLM calls."""
        sess = self._csess(user_id)
        pairs = sess.get_pairs()
        if not pairs:
            return None

        # Cache key: number of pairs — regenerate only when new turns are added
        cache_key = f"{user_id}_live"
        cached = getattr(self, '_live_summary_cache', {})
        cached_count = getattr(self, '_live_summary_count', {})

        if (cache_key in cached
                and cached_count.get(cache_key, 0) == len(pairs)):
            return cached[cache_key]

        # Build a quick summary from pairs without LLM (instant, no API call)
        # Summarise what the user discussed
        user_topics = []
        for h, _a in pairs:
            snippet = h.strip()[:80]
            if snippet:
                user_topics.append(snippet)

        title = "Current conversation"
        if user_topics:
            # Use the first user message as the title hint
            title = user_topics[0][:60]
            if len(title) >= 60:
                title = title[:57] + "…"

        bullets = []
        for topic in user_topics[:3]:
            bullets.append(topic[:120])

        live = {
            "date": datetime.now(timezone.utc).strftime("%b %d, %Y"),
            "title": f" {title}",
            "bullets": bullets,
            "thread_id": "__live__",
        }

        # Cache it
        if not hasattr(self, '_live_summary_cache'):
            self._live_summary_cache = {}
            self._live_summary_count = {}
        self._live_summary_cache[cache_key] = live
        self._live_summary_count[cache_key] = len(pairs)

        return live

    def get_kg_edges(self, user_id: str) -> List[Dict[str, str]]:
        """Return user-specific knowledge graph triples."""
        kg = self._get_kg(user_id)
        return [
            {"subject": e.source, "relation": e.relation, "object": e.target}
            for e in kg.edges
        ]

    def get_global_facts(self) -> List[Dict[str, str]]:
        """Return all global summary memory facts."""
        return [
            {"text": f.text, "source_user": f.source_user, "created_at": f.created_at}
            for f in self._global_summary.all_facts()
        ]

    def get_global_kg_edges(self) -> List[Dict[str, str]]:
        """Return all global knowledge graph triples."""
        return [
            {"subject": e.source, "relation": e.relation, "object": e.target}
            for e in self._global_kg.edges
        ]

    def get_session_metadata(self, user_id: str) -> Dict[str, Any]:
        """Return session metadata for display."""
        meta = self._smeta(user_id)
        n_conv = self._conv_counts.get(user_id, 0)
        n_msg  = self._msg_counts.get(user_id, 0)
        return {
            "user_id": meta.user_id,
            "session_id": meta.session_id,
            "subscription_tier": meta.subscription_tier,
            "session_start": meta.session_start,
            "total_conversations": n_conv,
            "total_messages": n_msg,
            "avg_messages_per_conv": (n_msg / n_conv) if n_conv else 0.0,
            "last_active": meta.last_active,
        }