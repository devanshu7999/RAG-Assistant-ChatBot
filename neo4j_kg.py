"""
neo4j_kg.py  -  Neo4j-backed Knowledge Graph classes
=====================================================

Drop-in replacements for the in-memory KnowledgeGraph (per-user)
and GlobalKnowledgeGraph (per-user account-level) that persist triples to Neo4j.

Graph Model
-----------
  (:Concept {name: "python"})
      -[:RELATES_TO {relation: "is_a", user_id: "dev", scope: "user"}]->
  (:Concept {name: "programming language"})

Visualise everything in Neo4j Browser:
  MATCH (a)-[r:RELATES_TO]->(b) RETURN a, r, b

Filter by user (conversation-level KG):
  MATCH (a)-[r:RELATES_TO {user_id: "devanshu", scope: "user"}]->(b) RETURN a, r, b

Filter by user (account-level KG):
  MATCH (a)-[r:RELATES_TO {user_id: "devanshu", scope: "account"}]->(b) RETURN a, r, b
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


# ─────────────────────────────────────────────────────────────────────────────
# Per-user Neo4j Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jKnowledgeGraph:
    """
    Neo4j-backed per-user Knowledge Graph.

    Same public API as the in-memory KnowledgeGraph in memory_manager.py:
      add(), query(), extract_and_store(), stats()
    """

    _EXTRACT_PROMPT = (
        "Extract knowledge graph triples from the text below.\n"
        "Return ONLY a JSON array of objects with keys: "
        '"subject", "relation", "object".\n'
        "Keep each value ≤ 5 words. Output NOTHING else.\n\n"
        "TEXT:\n{text}"
    )

    def __init__(self, driver, user_id: str):
        self._driver = driver
        self._user_id = user_id
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """Create index on Concept.name for fast lookups (idempotent)."""
        try:
            with self._driver.session() as session:
                session.run(
                    "CREATE INDEX concept_name_idx IF NOT EXISTS "
                    "FOR (c:Concept) ON (c.name)"
                )
        except Exception as exc:
            print(f"[Neo4jKG] Could not create index: {exc}")

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(self, subject: str, relation: str, obj: str) -> None:
        """MERGE a triple into Neo4j for this user."""
        s, o = subject.lower().strip(), obj.lower().strip()
        if not s or not relation.strip() or not o:
            return
        try:
            with self._driver.session() as session:
                session.run(
                    """
                    MERGE (a:Concept {name: $source})
                    MERGE (b:Concept {name: $target})
                    MERGE (a)-[r:RELATES_TO {relation: $relation,
                                             user_id: $user_id,
                                             scope: 'user'}]->(b)
                    """,
                    source=s, target=o, relation=relation.strip(),
                    user_id=self._user_id,
                )
        except Exception as exc:
            print(f"[Neo4jKG.add] {exc}")

    def extract_and_store(self, text: str, llm: ChatGroq,
                          max_triples: int = 8) -> None:
        """Ask the LLM for triples and store them in Neo4j."""
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

    # ── Queries ───────────────────────────────────────────────────────────────

    def query(self, tokens: List[str]) -> str:
        """Return matching triples as a formatted prompt block."""
        if not tokens:
            return ""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Concept)-[r:RELATES_TO {user_id: $user_id, scope: 'user'}]->(b:Concept)
                    WHERE any(tok IN $tokens WHERE a.name CONTAINS tok OR b.name CONTAINS tok)
                    RETURN a.name AS source, r.relation AS relation, b.name AS target
                    LIMIT 15
                    """,
                    user_id=self._user_id, tokens=tokens,
                )
                hits = [f"{rec['source']} {rec['relation']} {rec['target']}"
                        for rec in result]
            if not hits:
                return ""
            return "[Knowledge Graph]\n" + "\n".join(f"• {h}" for h in hits)
        except Exception as exc:
            print(f"[Neo4jKG.query] {exc}")
            return ""

    def stats(self) -> Dict[str, int]:
        """Return node/edge counts for this user."""
        try:
            with self._driver.session() as session:
                edge_count = session.run(
                    "MATCH ()-[r:RELATES_TO {user_id: $user_id, scope: 'user'}]->() "
                    "RETURN count(r) AS cnt",
                    user_id=self._user_id,
                ).single()["cnt"]
                # Nodes connected to this user's edges
                node_count = session.run(
                    """
                    MATCH (a)-[r:RELATES_TO {user_id: $user_id, scope: 'user'}]->(b)
                    WITH collect(DISTINCT a) + collect(DISTINCT b) AS nodes
                    UNWIND nodes AS n
                    RETURN count(DISTINCT n) AS cnt
                    """,
                    user_id=self._user_id,
                ).single()["cnt"]
            return {"nodes": node_count, "edges": edge_count}
        except Exception as exc:
            print(f"[Neo4jKG.stats] {exc}")
            return {"nodes": 0, "edges": 0}

    @property
    def edges(self) -> list:
        """Return all edges as a list of objects with .source, .relation, .target."""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Concept)-[r:RELATES_TO {user_id: $user_id, scope: 'user'}]->(b:Concept)
                    RETURN a.name AS source, r.relation AS relation, b.name AS target
                    """,
                    user_id=self._user_id,
                )
                return [_EdgeRecord(rec["source"], rec["relation"], rec["target"])
                        for rec in result]
        except Exception as exc:
            print(f"[Neo4jKG.edges] {exc}")
            return []

    @property
    def nodes(self) -> Dict[str, str]:
        """Return all nodes connected to this user's edges."""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a)-[r:RELATES_TO {user_id: $user_id, scope: 'user'}]->(b)
                    WITH collect(DISTINCT a) + collect(DISTINCT b) AS nodes
                    UNWIND nodes AS n
                    RETURN n.name AS name
                    """,
                    user_id=self._user_id,
                )
                return {rec["name"]: "Concept" for rec in result}
        except Exception as exc:
            print(f"[Neo4jKG.nodes] {exc}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-User Account-Level Neo4j Knowledge Graph
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jGlobalKnowledgeGraph:
    """
    Neo4j-backed per-user account-level Knowledge Graph.

    Each user has their own isolated KG that persists across all their
    conversations.  Triples are scoped with ``scope='account'`` and
    filtered by ``user_id`` — no cross-user data leakage.

    Same public API as GlobalKnowledgeGraph in global_memory.py:
      add(), query(), extract_and_store(), stats()
    """

    _EXTRACT_PROMPT = (
        "Extract knowledge graph triples about DOMAIN CONCEPTS (not personal user info) "
        "from the text below.\n"
        "Return ONLY a JSON array of objects with keys: "
        '"subject", "relation", "object".\n'
        "Keep each value ≤ 5 words. Output NOTHING else.\n\n"
        "TEXT:\n{text}"
    )

    def __init__(self, driver, user_id: str):
        self._driver = driver
        self._user_id = user_id

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add(self, subject: str, relation: str, obj: str) -> None:
        """MERGE an account-level triple into Neo4j for this user."""
        s, o = subject.lower().strip(), obj.lower().strip()
        if not s or not relation.strip() or not o:
            return
        try:
            with self._driver.session() as session:
                session.run(
                    """
                    MERGE (a:Concept {name: $source})
                    MERGE (b:Concept {name: $target})
                    MERGE (a)-[r:RELATES_TO {relation: $relation,
                                             user_id: $user_id,
                                             scope: 'account'}]->(b)
                    """,
                    source=s, target=o, relation=relation.strip(),
                    user_id=self._user_id,
                )
        except Exception as exc:
            print(f"[Neo4jGlobalKG.add] {exc}")

    def extract_and_store(self, text: str, llm: ChatGroq,
                          max_triples: int = 8) -> None:
        """Ask the LLM for domain triples and store in this user's account KG."""
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

    # ── Queries ───────────────────────────────────────────────────────────────

    def query(self, tokens: List[str]) -> str:
        """Return matching account-level triples for this user."""
        if not tokens:
            return ""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Concept)-[r:RELATES_TO {user_id: $user_id, scope: 'account'}]->(b:Concept)
                    WHERE any(tok IN $tokens WHERE a.name CONTAINS tok OR b.name CONTAINS tok)
                    RETURN a.name AS source, r.relation AS relation, b.name AS target
                    LIMIT 15
                    """,
                    user_id=self._user_id, tokens=tokens,
                )
                hits = [f"{rec['source']} {rec['relation']} {rec['target']}"
                        for rec in result]
            if not hits:
                return ""
            return "[Account Knowledge Graph]\n" + "\n".join(f"• {h}" for h in hits)
        except Exception as exc:
            print(f"[Neo4jGlobalKG.query] {exc}")
            return ""

    def stats(self) -> Dict[str, int]:
        """Return node/edge counts for this user's account KG."""
        try:
            with self._driver.session() as session:
                edge_count = session.run(
                    "MATCH ()-[r:RELATES_TO {user_id: $user_id, scope: 'account'}]->() "
                    "RETURN count(r) AS cnt",
                    user_id=self._user_id,
                ).single()["cnt"]
                node_count = session.run(
                    """
                    MATCH (a)-[r:RELATES_TO {user_id: $user_id, scope: 'account'}]->(b)
                    WITH collect(DISTINCT a) + collect(DISTINCT b) AS nodes
                    UNWIND nodes AS n
                    RETURN count(DISTINCT n) AS cnt
                    """,
                    user_id=self._user_id,
                ).single()["cnt"]
            return {"nodes": node_count, "edges": edge_count}
        except Exception as exc:
            print(f"[Neo4jGlobalKG.stats] {exc}")
            return {"nodes": 0, "edges": 0}

    @property
    def edges(self) -> list:
        """Return all edges for this user's account KG."""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a:Concept)-[r:RELATES_TO {user_id: $user_id, scope: 'account'}]->(b:Concept)
                    RETURN a.name AS source, r.relation AS relation, b.name AS target
                    """,
                    user_id=self._user_id,
                )
                return [_EdgeRecord(rec["source"], rec["relation"], rec["target"])
                        for rec in result]
        except Exception as exc:
            print(f"[Neo4jGlobalKG.edges] {exc}")
            return []

    @property
    def nodes(self) -> Dict[str, str]:
        """Return all nodes for this user's account KG."""
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (a)-[r:RELATES_TO {user_id: $user_id, scope: 'account'}]->(b)
                    WITH collect(DISTINCT a) + collect(DISTINCT b) AS nodes
                    UNWIND nodes AS n
                    RETURN n.name AS name
                    """,
                    user_id=self._user_id,
                )
                return {rec["name"]: "Concept" for rec in result}
        except Exception as exc:
            print(f"[Neo4jGlobalKG.nodes] {exc}")
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: lightweight edge record (mimics KGEdge / GlobalKGEdge dataclass)
# ─────────────────────────────────────────────────────────────────────────────

class _EdgeRecord:
    """Mimics the KGEdge dataclass so existing code like get_kg_edges() works."""
    __slots__ = ("source", "relation", "target")

    def __init__(self, source: str, relation: str, target: str):
        self.source = source
        self.relation = relation
        self.target = target
