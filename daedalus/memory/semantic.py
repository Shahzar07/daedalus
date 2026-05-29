"""Semantic memory — recall by *meaning*, not just shared keywords.

FTS recall (:mod:`daedalus.memory.store`) matches words; it misses *"my car won't
start"* ↔ *"the vehicle's engine is dead."* This index embeds each stored fact into a
vector with a small ``sentence-transformers`` model and recalls by cosine similarity, so
paraphrases still surface.

It's the one genuinely heavy upgrade (a model download on first use), so it lives behind
the optional ``[semantic]`` extra and is **off by default**. Everything degrades to a
clean no-op when the library is absent or ``SEMANTIC_MEMORY`` is false — :meth:`available`
gates every call, the model loads lazily on first real use, and vectors are cached in the
same ``state.db`` so they're computed once. Wire it on with::

    uv pip install -e ".[semantic]"     # then set SEMANTIC_MEMORY=true in .env
"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path

# Small, fast, widely-cached model — good quality for short facts, ~80MB.
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


class SemanticIndex:
    """Vector recall over stored memories. A no-op unless enabled *and* installed.

    Construct it alongside :class:`MemoryStore` (same db path). Call :meth:`add` whenever a
    fact is remembered and :meth:`search` to recall by meaning. Both are safe to call
    unconditionally — they return early when :meth:`available` is false.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        enabled: bool = False,
        model_name: str = _DEFAULT_MODEL,
    ):
        self.enabled = enabled
        self.model_name = model_name
        self._model = None  # lazily constructed on first real use
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS embeddings(content TEXT, vec BLOB)")
        self.conn.commit()

    def available(self) -> bool:
        """True only when enabled by config *and* the optional library imports."""
        if not self.enabled:
            return False
        try:
            import numpy  # noqa: F401
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _model_or_none(self):
        """Load (and cache) the embedding model, or return None if unavailable."""
        if not self.available():
            return None
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, text: str) -> list[float] | None:
        model = self._model_or_none()
        if model is None:
            return None
        vec = model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]

    # ---- writes --------------------------------------------------------------

    def add(self, content: str) -> bool:
        """Embed and store one memory. Returns False (no-op) when unavailable."""
        content = content.strip()
        if not content:
            return False
        vec = self._embed(content)
        if vec is None:
            return False
        with self._lock:
            self.conn.execute(
                "INSERT INTO embeddings(content, vec) VALUES (?, ?)", (content, _pack(vec))
            )
            self.conn.commit()
        return True

    # ---- reads ---------------------------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Return the stored memories most similar in meaning to ``query`` (or ``[]``)."""
        qvec = self._embed(query)
        if qvec is None:
            return []
        import numpy as np

        rows = self.conn.execute("SELECT content, vec FROM embeddings").fetchall()
        if not rows:
            return []
        q = np.asarray(qvec, dtype="float32")
        scored: list[tuple[float, str]] = []
        for content, blob in rows:
            v = np.asarray(_unpack(blob), dtype="float32")
            # Vectors are stored L2-normalized, so a dot product *is* cosine similarity.
            scored.append((float(np.dot(q, v)), content))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [content for _score, content in scored[:limit]]

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
