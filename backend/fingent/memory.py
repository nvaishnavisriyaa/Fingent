"""
Long-term memory — a vector store the agent reads from and writes to ACROSS sessions.

This is REAL semantic recall, not keyword overlap. Recall ranks stored memories by the cosine
similarity of their EMBEDDINGS, and the embedding is produced by a real model behind one
interface:

  * OpenAICompatibleEmbedder — a hosted embedding model over the OpenAI `/embeddings` API
    (OpenAI, Azure OpenAI, Cohere-compatible gateways, or a LOCAL sentence-transformers server
    exposed through text-embeddings-inference / Ollama / llama.cpp). Configured by
    FINGENT_EMBED_API_KEY (+ FINGENT_EMBED_BASE_URL, FINGENT_EMBED_MODEL).
  * SentenceTransformerEmbedder — an in-process `sentence-transformers` model when that package
    is installed (FINGENT_EMBED_BACKEND=sentence-transformers, FINGENT_EMBED_MODEL=<st-model>).
    Real local embeddings, no API key, no per-call network.
  * HashingEmbedder — a deterministic bag-of-tokens fallback used ONLY when no model is
    configured. It is LEXICAL (token overlap), clearly flagged `semantic=False`, and exists so the
    platform and the test suite still run fully offline. It is not presented as semantic recall.

Storage backends, all behind the same {add, recall} interface and all tenant-scoped:
  * SqlVectorMemory — DURABLE: rows live in the platform store (SQLite/Postgres), so memory
    survives a restart and tenant isolation is a SQL WHERE clause. The default when a store is
    available.
  * PineconeMemory — used when PINECONE_API_KEY is set; durable, cross-process, real ANN.
  * LocalVectorMemory — in-process cosine store; the no-store dev/offline default.

Each memory records the embedder that produced its vector, so recall never compares vectors from
different models (changing the embedder simply stops matching old rows instead of returning
garbage). Namespacing is per (tenant, agent) everywhere.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
import time

_log = logging.getLogger("fingent.memory")

EMBED_DIM = int(os.getenv("FINGENT_EMBED_DIM", "512"))   # hashing fallback dimension
_TOKEN = re.compile(r"[a-z0-9]+")


def _l2(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# Embedders — one interface, real model when configured, lexical fallback otherwise
# --------------------------------------------------------------------------- #
def _stable_hash(s: str) -> int:
    """A STABLE hash (blake2b) — unlike builtin hash(), it is identical across processes, so
    hashing-embedded vectors are reproducible and survive a restart (builtin hash() is salted)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder:
    """Deterministic bag-of-tokens + char-trigram STABLE hash, L2-normalized. LEXICAL, not semantic
    — similarity is token overlap. Offline/test fallback only (clearly flagged)."""
    semantic = False

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim
        self.name = f"hashing-lexical:{dim}"

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall((text or "").lower()):
            vec[_stable_hash("fingent\x00" + tok) % self.dim] += 1.0
        t = (text or "").lower()
        for i in range(len(t) - 2):
            vec[_stable_hash("tri\x00" + t[i:i + 3]) % self.dim] += 0.5
        return _l2(vec)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class OpenAICompatibleEmbedder:
    """Hosted embeddings over the OpenAI-compatible `/embeddings` API. Provider-agnostic: point
    FINGENT_EMBED_BASE_URL at OpenAI, Azure OpenAI, a Cohere-compatible gateway, or a local
    sentence-transformers server (text-embeddings-inference / Ollama / llama.cpp)."""
    semantic = True

    def __init__(self) -> None:
        self.api_key = (os.getenv("FINGENT_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
        self.base_url = (os.getenv("FINGENT_EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("FINGENT_EMBED_MODEL") or "text-embedding-3-small"
        self.timeout = float(os.getenv("FINGENT_EMBED_TIMEOUT", "30"))
        self.name = f"openai:{self.model}"
        self._dim = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        import requests
        if not texts:
            return []
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "input": list(texts)}, timeout=self.timeout)
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d.get("index", 0))
        vecs = [_l2([float(x) for x in d["embedding"]]) for d in data]
        if vecs:
            self._dim = len(vecs[0])
        return vecs

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed_one("dimension probe")
        return self._dim or 0


class SentenceTransformerEmbedder:
    """In-process `sentence-transformers` model (real local embeddings, no API). Lazy-loaded."""
    semantic = True

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: only when selected
        self.model_name = os.getenv("FINGENT_EMBED_MODEL") or "all-MiniLM-L6-v2"
        self._model = SentenceTransformer(self.model_name)
        self.name = f"st:{self.model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]

    def embed_one(self, text: str) -> list[float]:
        return self.embed_many([text])[0]


_EMBEDDER = None
_EMBED_LOCK = threading.Lock()


def _build_embedder():
    backend = (os.getenv("FINGENT_EMBED_BACKEND") or "").lower().strip()
    if backend in ("hashing", "lexical"):
        return HashingEmbedder()
    if backend in ("sentence-transformers", "sentence_transformers", "st"):
        try:
            return SentenceTransformerEmbedder()
        except Exception as e:  # noqa: BLE001
            _log.warning("sentence-transformers requested but unavailable (%s); "
                         "falling back to lexical hashing memory", e)
            return HashingEmbedder()
    emb = OpenAICompatibleEmbedder()
    if emb.enabled:
        return emb
    # nothing configured: lexical fallback (NOT presented as semantic recall)
    return HashingEmbedder()


def get_embedder():
    """Process-wide embedder. Real model when configured (FINGENT_EMBED_API_KEY or
    sentence-transformers); deterministic lexical hashing otherwise."""
    global _EMBEDDER
    with _EMBED_LOCK:
        if _EMBEDDER is None:
            _EMBEDDER = _build_embedder()
        return _EMBEDDER


def set_embedder(embedder) -> None:
    """Override the active embedder (used by recall-quality tests to inject a known model)."""
    global _EMBEDDER
    with _EMBED_LOCK:
        _EMBEDDER = embedder


def embed(text: str) -> list[float]:
    """Embed text with the active embedder (back-compat module-level helper). Resilient: on a
    provider error it falls back to the lexical hashing embedder so a call never crashes."""
    try:
        return get_embedder().embed_one(text)
    except Exception as e:  # noqa: BLE001
        _log.warning("embedding failed (%s); using lexical fallback for this call", e)
        return HashingEmbedder().embed_one(text)


# --------------------------------------------------------------------------- #
# Storage backends — same {add, recall} interface, all tenant/agent-namespaced
# --------------------------------------------------------------------------- #
def _score_rows(query_vec, embedder_name, rows, k, min_score):
    scored = []
    for r in rows:
        if r.get("embedder") and r["embedder"] != embedder_name:
            continue                                  # never mix vectors from different models
        v = r.get("vec") or []
        if len(v) != len(query_vec):
            continue
        scored.append({"text": r["text"], "meta": r.get("meta", {}),
                       "score": round(_cosine(query_vec, v), 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scored if s["score"] >= min_score][:k]


class LocalVectorMemory:
    """In-process cosine vector store, namespaced by (tenant, agent). No-store dev/offline default
    (NOT durable across restarts — use SqlVectorMemory or Pinecone for persistence)."""

    def __init__(self) -> None:
        self._store = {}
        self._lock = threading.Lock()

    @property
    def backend(self) -> str:
        return "local"

    def add(self, tenant: str, agent: str, text: str, meta: dict | None = None) -> str:
        emb = get_embedder()
        vec = embed(text)
        mid = f"mem_{int(time.time()*1000)}_{abs(hash(text)) % 10000}"
        rec = {"id": mid, "text": text, "meta": meta or {}, "vec": vec,
               "embedder": emb.name, "ts": time.time()}
        with self._lock:
            self._store.setdefault((tenant, agent), []).append(rec)
        return mid

    def recall(self, tenant: str, agent: str, query: str, k: int = 3,
               min_score: float = 0.05) -> list[dict]:
        emb = get_embedder()
        try:
            q = emb.embed_one(query)
        except Exception:  # noqa: BLE001 — recall must never break the chat turn
            return []
        with self._lock:
            rows = list(self._store.get((tenant, agent), []))
        return _score_rows(q, emb.name, rows, k, min_score)


class SqlVectorMemory:
    """DURABLE vector memory backed by the platform store (SQLite/Postgres). Survives a restart;
    tenant isolation is a SQL WHERE clause. State lives in the DB, so this object is stateless
    and cheap to construct."""

    def __init__(self, store) -> None:
        self.store = store

    @property
    def backend(self) -> str:
        return "sql"

    def add(self, tenant: str, agent: str, text: str, meta: dict | None = None) -> str:
        emb = get_embedder()
        vec = embed(text)
        mid = f"mem_{int(time.time()*1000)}_{abs(hash(text)) % 100000}"
        self.store.add_memory(tenant, agent, mid, text, meta or {}, emb.name, len(vec), vec)
        return mid

    def recall(self, tenant: str, agent: str, query: str, k: int = 3,
               min_score: float = 0.05) -> list[dict]:
        emb = get_embedder()
        try:
            q = emb.embed_one(query)
        except Exception:  # noqa: BLE001
            return []
        rows = self.store.list_memories(tenant, agent)
        return _score_rows(q, emb.name, rows, k, min_score)


class PineconeMemory:
    """Pinecone-backed durable memory. Vectors come from the configured embedder; the index is
    created at the embedder's dimensionality. Namespaced per (tenant, agent)."""

    def __init__(self) -> None:
        from pinecone import Pinecone, ServerlessSpec  # lazy: only when a key is configured
        self._index_name = os.getenv("PINECONE_INDEX", "fingent-memory")
        self._cloud = os.getenv("PINECONE_CLOUD", "aws")
        self._region = os.getenv("PINECONE_REGION", "us-east-1")
        self._dim = get_embedder().dim if hasattr(get_embedder(), "dim") else EMBED_DIM
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing = {i["name"] for i in pc.list_indexes()}
        if self._index_name not in existing:
            pc.create_index(name=self._index_name, dimension=self._dim, metric="cosine",
                            spec=ServerlessSpec(cloud=self._cloud, region=self._region))
        self._index = pc.Index(self._index_name)

    @property
    def backend(self) -> str:
        return "pinecone"

    @staticmethod
    def _ns(tenant: str, agent: str) -> str:
        return f"{tenant}:{agent}"

    def add(self, tenant: str, agent: str, text: str, meta: dict | None = None) -> str:
        emb = get_embedder()
        mid = f"mem_{int(time.time()*1000)}_{abs(hash(text)) % 100000}"
        md = {"text": text[:1500], "embedder": emb.name,
              **{k: str(v)[:200] for k, v in (meta or {}).items()}}
        self._index.upsert(vectors=[{"id": mid, "values": embed(text), "metadata": md}],
                           namespace=self._ns(tenant, agent))
        return mid

    def recall(self, tenant: str, agent: str, query: str, k: int = 3,
               min_score: float = 0.05) -> list[dict]:
        emb = get_embedder()
        try:
            qv = emb.embed_one(query)
        except Exception:  # noqa: BLE001
            return []
        res = self._index.query(vector=qv, top_k=k, include_metadata=True,
                                namespace=self._ns(tenant, agent))
        out = []
        for m in (res.get("matches") if isinstance(res, dict) else res.matches) or []:
            md = (m.get("metadata") if isinstance(m, dict) else m.metadata) or {}
            score = (m.get("score") if isinstance(m, dict) else m.score) or 0.0
            if md.get("embedder") and md["embedder"] != emb.name:
                continue
            if score >= min_score:
                out.append({"text": md.get("text", ""), "meta": md, "score": round(score, 4)})
        return out


_MEMORY = None
_MEMORY_STORE_ID = None
_MEM_LOCK = threading.Lock()


def get_memory(store=None):
    """Return the process-wide memory backend.

    Selection: Pinecone if PINECONE_API_KEY is set; else a DURABLE SqlVectorMemory bound to the
    provided platform store; else the in-process LocalVectorMemory (no-store dev/offline default).
    Passing a store upgrades memory from in-process to persistent without changing callers.
    """
    global _MEMORY, _MEMORY_STORE_ID
    with _MEM_LOCK:
        if os.getenv("PINECONE_API_KEY"):
            if not isinstance(_MEMORY, PineconeMemory):
                try:
                    _MEMORY = PineconeMemory()
                except Exception as e:  # noqa: BLE001 — never break the platform on memory init
                    _log.warning("Pinecone init failed (%s); falling back to local memory", e)
                    _MEMORY = LocalVectorMemory()
            return _MEMORY
        if store is not None:
            sid = id(store)
            if not isinstance(_MEMORY, SqlVectorMemory) or _MEMORY_STORE_ID != sid:
                _MEMORY = SqlVectorMemory(store)
                _MEMORY_STORE_ID = sid
            return _MEMORY
        if _MEMORY is None:
            _MEMORY = LocalVectorMemory()
        return _MEMORY


def reset_memory_for_tests() -> None:
    global _MEMORY, _MEMORY_STORE_ID, _EMBEDDER
    with _MEM_LOCK:
        _MEMORY = None
        _MEMORY_STORE_ID = None
    with _EMBED_LOCK:
        _EMBEDDER = None
