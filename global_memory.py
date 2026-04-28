"""
global_memory.py  –  Per-user account-level memory layers
=======================================================================

Two layers:
  1. GlobalSummaryMemory        – condensed knowledge facts from all interactions
                                  (shared across all users; general/domain knowledge only)
  2. UserAccountKnowledgeGraph  – per-user account-level entity-relation triples
                                  (isolated per user, shared across all their chats;
                                   no cross-user data leakage)

GlobalSummaryMemory is injected into every prompt for every user.
UserAccountKnowledgeGraph is per-user: each user’s KG is private and persists
across all their conversations.  Guests get no KG access.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from app_config import MAX_GLOBAL_FACTS


# ─────────────────────────────────────────────────────────────────────────────
# Global Summary Memory
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalFact:
    """A single condensed knowledge fact shared across all users."""
    text:       str
    source_user: str  = "system"     # which user's interaction produced this
    created_at: str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fact_id:    str   = field(default_factory=lambda: uuid.uuid4().hex[:6])


class GlobalSummaryMemory:
    """
    Stores condensed knowledge from ALL past interactions across ALL users.

    When any conversation is closed or an exchange yields a durable fact,
    it gets distilled into this shared store.  Every prompt — regardless
    of user — receives these facts as context.

    Design principle: accumulate general knowledge, not user-private data.
    The extraction prompt filters for general/domain knowledge only.
    """

    _EXTRACT_PROMPT = (
        "You are a knowledge distillation assistant.\n"
        "Read the exchange below and decide if there is ONE general, "
        "domain-specific knowledge fact worth storing for ALL users "
        "(e.g. a technical concept, a definition, a best practice, "
        "a factual claim from a document).\n"
        "DO NOT extract personal user preferences or private information.\n"
        "If yes, output ONLY the fact as a single sentence.\n"
        "If no useful general fact exists, output exactly: NONE\n\n"
        "Exchange:\nUser: {human}\nAssistant: {assistant}"
    )

    _SUMMARISE_EXTRACT_PROMPT = (
        "You are a knowledge distillation assistant.\n"
        "Read the conversation summary below and extract up to 2 general "
        "domain-specific knowledge facts worth storing for ALL future users.\n"
        "DO NOT extract personal user preferences or private information.\n"
        "Return ONLY a JSON array of strings. If no facts, return []\n\n"
        "Summary:\nTitle: {title}\nBullets:\n{bullets}"
    )

    def __init__(self):
        self._facts: List[GlobalFact] = []

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add_fact(self, text: str, source_user: str = "system") -> Optional[GlobalFact]:
        """Add a global fact; deduplicate near-identical entries."""
        text = text.strip()
        if not text or text.upper() == "NONE":
            return None
        for f in self._facts:
            if f.text.lower() == text.lower():
                return f  # already known
        fact = GlobalFact(text=text, source_user=source_user)
        self._facts.append(fact)
        # Cap: drop oldest facts when exceeding limit
        while len(self._facts) > MAX_GLOBAL_FACTS:
            self._facts.pop(0)
        return fact

    # ── Auto-extraction from exchange ─────────────────────────────────────────

    def try_extract_from_exchange(
        self, human: str, assistant: str, llm: ChatGroq, source_user: str = "system"
    ) -> Optional[str]:
        """Ask the LLM if this exchange contains a general knowledge fact."""
        prompt = self._EXTRACT_PROMPT.format(human=human[:800], assistant=assistant[:800])
        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if raw.upper() == "NONE" or not raw:
                return None
            if "?" not in raw and len(raw) > 10:
                self.add_fact(raw, source_user=source_user)
                return raw
        except Exception as exc:
            print(f"[GlobalSummaryMemory.try_extract_from_exchange] {exc}")
        return None

    def try_extract_from_summary(
        self, title: str, bullets: List[str], llm: ChatGroq, source_user: str = "system"
    ) -> List[str]:
        """Extract general facts from a conversation summary."""
        bullets_text = "\n".join(f"- {b}" for b in bullets)
        prompt = self._SUMMARISE_EXTRACT_PROMPT.format(title=title, bullets=bullets_text)
        extracted: List[str] = []
        try:
            raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            facts = json.loads(raw)
            for fact_text in facts:
                if isinstance(fact_text, str) and fact_text.strip():
                    result = self.add_fact(fact_text.strip(), source_user=source_user)
                    if result:
                        extracted.append(fact_text.strip())
        except Exception as exc:
            print(f"[GlobalSummaryMemory.try_extract_from_summary] {exc}")
        return extracted

    # ── Prompt block ──────────────────────────────────────────────────────────

    def to_prompt_block(self) -> str:
        if not self._facts:
            return ""
        lines = ["[Global Knowledge (shared across all sessions)]"]
        for f in self._facts:
            lines.append(f"• {f.text}")
        return "\n".join(lines)

    def all_facts(self) -> List[GlobalFact]:
        return list(self._facts)

    def __len__(self) -> int:
        return len(self._facts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-User Account-Level Knowledge Graph (in-memory fallback)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GlobalKGEdge:
    source:   str
    relation: str
    target:   str


class UserAccountKnowledgeGraph:
    """
    Per-user account-level Knowledge Graph (in-memory fallback when Neo4j
    is unavailable).  Each user gets their own isolated instance that
    persists across all their conversations.  No cross-user data leakage;
    guests are never given access to this layer.

    Stores structured relationships between domain entities and concepts.
    Separate from per-conversation KG: this captures account-wide
    knowledge, not per-exchange personal facts.
    """

    _EXTRACT_PROMPT = (
        "Extract knowledge graph triples about DOMAIN CONCEPTS (not personal user info) "
        "from the text below.\n"
        "Return ONLY a JSON array of objects with keys: "
        '"subject", "relation", "object".\n'
        "Keep each value ≤ 5 words. Output NOTHING else.\n\n"
        "TEXT:\n{text}"
    )

    def __init__(self):
        self.nodes: Dict[str, str] = {}       # entity → type
        self.edges: List[GlobalKGEdge] = []

    def add(self, subject: str, relation: str, obj: str) -> None:
        s, o = subject.lower().strip(), obj.lower().strip()
        self.nodes.setdefault(s, "Concept")
        self.nodes.setdefault(o, "Concept")
        if not any(
            e.source == s and e.relation == relation and e.target == o
            for e in self.edges
        ):
            self.edges.append(GlobalKGEdge(s, relation, o))

    def extract_and_store(
        self, text: str, llm: ChatGroq, max_triples: int = 8
    ) -> None:
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
        """Query the global KG for triples matching any of the given tokens."""
        hits = []
        for e in self.edges:
            if any(tok in e.source or tok in e.target for tok in tokens):
                hits.append(f"{e.source} {e.relation} {e.target}")
        if not hits:
            return ""
        unique = list(dict.fromkeys(hits))[:15]
        return "[Account Knowledge Graph]\n" + "\n".join(f"• {h}" for h in unique)

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}
