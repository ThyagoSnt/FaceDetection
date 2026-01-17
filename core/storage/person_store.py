from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import os
import sqlite3
import threading

import numpy as np
import faiss


@dataclass
class MatchResult:
    person_id: str | None
    score: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(x)  # in-place
    return x


class PersonStore:
    """
    SQLite (source of truth) + FAISS (fast similarity search).

    - Stores multiple embeddings per person_id.
    - Uses cosine similarity by normalizing vectors and searching with inner product.
    """

    def __init__(
        self,
        db_path: str = "/data/person_store.sqlite",
        index_path: str = "/data/person_store.faiss",
        top_k: int = 10,
        persist_index: bool = True,
    ) -> None:
        self.db_path = db_path
        self.index_path = index_path
        self.top_k = int(top_k)
        self.persist_index = bool(persist_index)

        self._lock = threading.Lock()
        self._conn = self._open_db()
        self._ensure_schema()

        self._index: faiss.Index = self._load_or_build_index()

    # DB
    def _open_db(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                vec_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id  TEXT NOT NULL,
                dim        INTEGER NOT NULL,
                vec        BLOB NOT NULL
            );
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_person ON embeddings(person_id);")
        self._conn.commit()

    # FAISS
    def _make_empty_index(self, dim: int) -> faiss.Index:
        # Cosine similarity = inner product over L2-normalized vectors
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    def _load_or_build_index(self) -> faiss.Index:
        """
        Strategy:
        - If a FAISS index exists, load it (fast startup).
        - If not, rebuild from SQLite.
        - Even if loaded, we assume vec_id alignment is consistent with SQLite.
        """
        # Try load from disk first
        if os.path.isfile(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception:
                # If index file is corrupted/outdated, fall back to rebuild
                pass

        # Rebuild from DB
        cur = self._conn.execute("SELECT vec_id, dim, vec FROM embeddings ORDER BY vec_id ASC;")
        rows = cur.fetchall()
        if not rows:
            # Unknown dim yet; create a dummy 1D index that will be replaced on first add
            return self._make_empty_index(1)

        dim = int(rows[0][1])
        index = self._make_empty_index(dim)

        ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
        vecs = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows]).astype(np.float32)
        vecs = vecs.reshape(len(rows), dim)

        # Ensure normalized (in case old data wasn't)
        faiss.normalize_L2(vecs)
        index.add_with_ids(vecs, ids)

        if self.persist_index:
            self._save_index(index)

        return index

    def _save_index(self, index: faiss.Index) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        tmp = self.index_path + ".tmp"
        faiss.write_index(index, tmp)
        os.replace(tmp, self.index_path)

    def _ensure_index_dim(self, dim: int) -> None:
        """
        If index is still dummy (d=1 and empty) or mismatched, re-create it.
        """
        if self._index.ntotal == 0 and self._index.d == 1 and dim != 1:
            self._index = self._make_empty_index(dim)
            return

        if self._index.d != dim:
            raise ValueError(f"Embedding dim mismatch: index_dim={self._index.d} vs emb_dim={dim}")

    # Public API (same spirit as yours)
    def add_embedding(self, person_id: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        dim = int(emb.shape[0])

        with self._lock:
            self._ensure_index_dim(dim)

            # Normalize and persist to SQLite
            emb_norm = _l2_normalize(emb).reshape(-1).astype(np.float32)
            blob = emb_norm.tobytes()

            cur = self._conn.execute(
                "INSERT INTO embeddings(person_id, dim, vec) VALUES (?, ?, ?);",
                (person_id, dim, sqlite3.Binary(blob)),
            )
            vec_id = int(cur.lastrowid)
            self._conn.commit()

            # Add to FAISS with the same vec_id
            x = emb_norm.reshape(1, dim).astype(np.float32)
            ids = np.array([vec_id], dtype=np.int64)
            self._index.add_with_ids(x, ids)

            if self.persist_index:
                self._save_index(self._index)

    def remove_person(self, person_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("SELECT vec_id FROM embeddings WHERE person_id = ?;", (person_id,))
            ids = [int(r[0]) for r in cur.fetchall()]
            if not ids:
                return False

            # Delete from SQLite
            self._conn.execute("DELETE FROM embeddings WHERE person_id = ?;", (person_id,))
            self._conn.commit()

            # Remove from FAISS
            ids_np = np.array(ids, dtype=np.int64)
            selector = faiss.IDSelectorBatch(ids_np)
            self._index.remove_ids(selector)  # supported by IDMap2 + Flat
            # (If you swap to other FAISS indexes later, check remove_ids support.)

            if self.persist_index:
                self._save_index(self._index)

            return True

    def has_any_persons(self) -> bool:
        cur = self._conn.execute("SELECT 1 FROM embeddings LIMIT 1;")
        return cur.fetchone() is not None

    def match_embedding(self, embedding: np.ndarray, threshold: float) -> MatchResult:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        dim = int(emb.shape[0])

        with self._lock:
            self._ensure_index_dim(dim)

            if self._index.ntotal == 0:
                return MatchResult(person_id=None, score=-1.0)

            q = _l2_normalize(emb)  # shape (1, d)
            k = max(1, self.top_k)

            D, I = self._index.search(q, k)  # inner product on normalized => cosine
            scores = D.reshape(-1).tolist()
            ids = I.reshape(-1).tolist()

        # Filter invalids
        pairs = [(int(vid), float(sc)) for vid, sc in zip(ids, scores) if int(vid) != -1]
        if not pairs:
            return MatchResult(person_id=None, score=-1.0)

        # Fetch person_id for returned vec_ids (single query)
        vec_ids = [p[0] for p in pairs]
        placeholders = ",".join("?" for _ in vec_ids)
        cur = self._conn.execute(
            f"SELECT vec_id, person_id FROM embeddings WHERE vec_id IN ({placeholders});",
            tuple(vec_ids),
        )
        id_to_person = {int(r[0]): str(r[1]) for r in cur.fetchall()}

        # Multiple vectors per person: keep best score per person_id
        best_person: str | None = None
        best_score: float = -1.0
        for vid, sc in pairs:
            pid = id_to_person.get(vid)
            if pid is None:
                continue
            if sc > best_score:
                best_score = sc
                best_person = pid

        if best_person is not None and best_score >= float(threshold):
            return MatchResult(person_id=best_person, score=best_score)
        return MatchResult(person_id=None, score=best_score)

    # Optional helper for your /persons endpoint (no need to introspect internals)
    def list_persons(self) -> List[Dict[str, int | str]]:
        cur = self._conn.execute(
            "SELECT person_id, COUNT(*) as n FROM embeddings GROUP BY person_id ORDER BY person_id ASC;"
        )
        return [{"person_id": str(pid), "num_images": int(n)} for (pid, n) in cur.fetchall()]
