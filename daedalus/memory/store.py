"""Long-term memory — SQLite with full-text recall.

Two stores live in one ``~/.dae/state.db``:

  * ``memories`` — short, curated facts the agent decides are worth keeping
    (written post-turn by the summarization step in the loop).
  * ``messages`` — an archive of every turn, so past conversation is searchable.

Recall is full-text via **FTS5**. FTS5 is a compile-time SQLite option; almost every
modern Python ships with it, but if a build lacks it we transparently fall back to a
plain table with ``LIKE`` matching. Either way the public API is identical, so the
rest of Daedalus never has to care.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {"the", "and", "for", "what", "with", "this", "that", "are", "was", "you", "your"}


def _tokens(text: str, limit: int = 8) -> list[str]:
    """Reduce free text to a few salient search tokens (lowercased, deduped)."""
    seen: list[str] = []
    for word in _TOKEN.findall(text.lower()):
        if len(word) >= 3 and word not in _STOPWORDS and word not in seen:
            seen.append(word)
        if len(seen) >= limit:
            break
    return seen


class MemoryStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: surfaces may touch the store from worker threads;
        # a lock serializes writes so that's safe.
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self.fts = self._init_schema()

    def _init_schema(self) -> bool:
        try:
            self.conn.executescript(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories "
                "USING fts5(content, kind UNINDEXED, ts UNINDEXED);"
                "CREATE VIRTUAL TABLE IF NOT EXISTS messages "
                "USING fts5(content, role UNINDEXED, run_id UNINDEXED, ts UNINDEXED);"
            )
            return True
        except sqlite3.OperationalError:
            # No FTS5 in this SQLite build — fall back to ordinary tables + LIKE.
            self.conn.executescript(
                "CREATE TABLE IF NOT EXISTS memories(content TEXT, kind TEXT, ts REAL);"
                "CREATE TABLE IF NOT EXISTS messages(content TEXT, role TEXT, run_id TEXT, ts REAL);"
            )
            return False

    # ---- writes --------------------------------------------------------------

    def remember(self, content: str, kind: str = "fact") -> None:
        content = content.strip()
        if not content:
            return
        with self._lock:
            self.conn.execute(
                "INSERT INTO memories(content, kind, ts) VALUES (?, ?, ?)",
                (content, kind, time.time()),
            )
            self.conn.commit()

    def archive_message(self, role: str, content: str, run_id: str) -> None:
        if not content:
            return
        with self._lock:
            self.conn.execute(
                "INSERT INTO messages(content, role, run_id, ts) VALUES (?, ?, ?, ?)",
                (content, role, run_id, time.time()),
            )
            self.conn.commit()

    # ---- reads ---------------------------------------------------------------

    def recall(self, query: str, limit: int = 5) -> list[str]:
        """Return curated memories most relevant to ``query`` (best-effort)."""
        tokens = _tokens(query)
        if not tokens:
            return []
        if self.fts:
            match = " OR ".join(f'"{t}"' for t in tokens)
            cur = self.conn.execute(
                "SELECT content FROM memories WHERE memories MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            )
        else:
            clause = " OR ".join("content LIKE ?" for _ in tokens)
            params: list[object] = [f"%{t}%" for t in tokens]
            params.append(limit)
            cur = self.conn.execute(
                f"SELECT content FROM memories WHERE {clause} ORDER BY ts DESC LIMIT ?", params
            )
        return [row[0] for row in cur.fetchall()]

    def recent(self, limit: int = 10) -> list[str]:
        cur = self.conn.execute("SELECT content FROM memories ORDER BY ts DESC LIMIT ?", (limit,))
        return [row[0] for row in cur.fetchall()]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM memories").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
