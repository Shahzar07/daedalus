"""Memory graph — turn flat facts into a tiny entity/relation network.

The FTS recall in :mod:`daedalus.memory.store` is great at "find facts containing
these words." A *graph* answers a different question: "what do I know that's
**connected** to this thing?" When a fact like *"Alice works at Acme"* is stored, we
also record the edge ``(alice) --works_at--> (acme)``. Later, a question that mentions
Alice *or* Acme can surface the connection even if the exact words don't match.

Extraction here is deliberately **lightweight and dependency-free** — a handful of
readable patterns over short curated facts, not an NLP pipeline. It runs in-process at
$0, which suits the teaching goal; the semantic index (:mod:`daedalus.memory.semantic`)
is the heavier, optional counterpart. Edges live in the *same* ``state.db`` as the rest
of memory, in their own table, so there's one file to back up and reason about.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path

# Relation verbs we recognize in a "<subject> <verb> <object>" fact. The value is the
# normalized edge label we store (so "works at" and "works in" both become "works_at").
_REL_VERBS: dict[str, str] = {
    "is": "is",
    "are": "is",
    "was": "is",
    "likes": "likes",
    "loves": "likes",
    "prefers": "prefers",
    "uses": "uses",
    "knows": "knows",
    "owns": "owns",
    "has": "has",
    "wants": "wants",
    "needs": "needs",
    "speaks": "speaks",
    "studies": "studies",
    "works": "works_at",
    "lives": "lives_in",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "and", "for", "with", "this", "that", "are", "was", "you", "your", "a", "an"}

# "Alice's favourite colour is blue"  ->  (alice, favourite_colour, blue)
_POSSESSIVE = re.compile(r"^(.+?)'s\s+([\w ]+?)\s+(?:is|are|was|=)\s+(.+)$", re.IGNORECASE)
# "Alice works at Acme" / "Bob is a teacher" -> subject, verb(+prep), object
_SVO = re.compile(r"^(.+?)\s+(\w+)\s+(.+)$")
# Prepositions we fold into the preceding verb so "works at X" -> works_at, object "X".
_PREPS = {"at", "in", "on", "for", "with", "to", "from", "a", "an", "the"}


def _norm(text: str) -> str:
    """Normalize an entity for storage/lookup: lowercased, trimmed, collapsed spaces."""
    return " ".join(text.lower().split()).strip(" .,:;!?")


def extract_triples(fact: str) -> list[tuple[str, str, str]]:
    """Pull ``(subject, relation, object)`` triples from one short fact (best-effort).

    Returns an empty list when nothing matches — most facts yield zero or one triple.
    Kept pure and import-free so it's trivially unit-testable.
    """
    fact = fact.strip()
    if not fact:
        return []

    m = _POSSESSIVE.match(fact)
    if m:
        subj, rel, obj = m.group(1), m.group(2).replace(" ", "_"), m.group(3)
        return [(_norm(subj), _norm(rel), _norm(obj))]

    m = _SVO.match(fact)
    if m:
        subj, verb, rest = m.group(1), m.group(2).lower(), m.group(3)
        rel = _REL_VERBS.get(verb)
        if rel:
            # Absorb a leading preposition/article into the object ("works at Acme").
            words = rest.split()
            while words and words[0].lower() in _PREPS:
                words.pop(0)
            obj = " ".join(words)
            if obj:
                return [(_norm(subj), rel, _norm(obj))]
    return []


def _tokens(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS]


class MemoryGraph:
    """A small persistent triple-store sharing the memory database.

    Public surface mirrors the rest of memory: :meth:`add_fact` to ingest, :meth:`related`
    to recall connected knowledge for a query, plus :meth:`neighbors`/:meth:`count` for
    inspection. All reads degrade to ``[]`` rather than raising, so a graph hiccup can
    never sink a turn.
    """

    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS graph_edges("
            "subject TEXT, relation TEXT, object TEXT, fact TEXT, ts REAL)"
        )
        self.conn.commit()

    # ---- writes --------------------------------------------------------------

    def add_fact(self, fact: str) -> int:
        """Extract triples from ``fact`` and store them. Returns how many edges were added."""
        triples = extract_triples(fact)
        if not triples:
            return 0
        with self._lock:
            for subj, rel, obj in triples:
                self.conn.execute(
                    "INSERT INTO graph_edges(subject, relation, object, fact, ts) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (subj, rel, obj, fact.strip(), time.time()),
                )
            self.conn.commit()
        return len(triples)

    # ---- reads ---------------------------------------------------------------

    def related(self, query: str, limit: int = 5) -> list[str]:
        """Return human-readable edges whose subject/object mentions a query token.

        Output like ``"alice works_at acme"`` — compact lines the loop folds into the
        prompt's recalled-memory section.
        """
        tokens = _tokens(query)
        if not tokens:
            return []
        clause = " OR ".join("(subject LIKE ? OR object LIKE ?)" for _ in tokens)
        params: list[object] = []
        for t in tokens:
            params.extend((f"%{t}%", f"%{t}%"))
        params.append(limit)
        try:
            cur = self.conn.execute(
                "SELECT DISTINCT subject, relation, object FROM graph_edges "
                f"WHERE {clause} ORDER BY ts DESC LIMIT ?",
                params,
            )
        except sqlite3.OperationalError:
            return []
        return [f"{s} {r} {o}" for s, r, o in cur.fetchall()]

    def neighbors(self, entity: str, limit: int = 10) -> list[tuple[str, str, str]]:
        """Return raw edges touching ``entity`` (as subject or object)."""
        e = f"%{_norm(entity)}%"
        cur = self.conn.execute(
            "SELECT subject, relation, object FROM graph_edges "
            "WHERE subject LIKE ? OR object LIKE ? ORDER BY ts DESC LIMIT ?",
            (e, e, limit),
        )
        return [(s, r, o) for s, r, o in cur.fetchall()]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM graph_edges").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
