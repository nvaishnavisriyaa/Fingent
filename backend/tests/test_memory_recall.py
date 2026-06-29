"""
Recall-QUALITY tests for long-term memory.

The point of these tests is to prove memory recalls by MEANING (embeddings), not by keyword
overlap, and that it is durable + tenant-scoped. Embeddings are deterministic and offline:
  * the lexical HashingEmbedder is the conftest default (token overlap), and
  * a small injected FakeSemanticEmbedder maps synonyms to a shared concept axis so we can show
    semantic recall succeeding where lexical recall fails — with NO shared words.

A real hosted-embedding eval (OpenAI-compatible) is included but SKIPPED unless an embedding key
is configured, so CI stays hermetic while the harness exists for a keyed environment.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent.store import Store
from fingent import memory as M
from fingent.memory import (
    HashingEmbedder, OpenAICompatibleEmbedder, LocalVectorMemory, SqlVectorMemory,
    get_embedder, set_embedder, reset_memory_for_tests,
)


def _l2(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class FakeSemanticEmbedder:
    """A tiny deterministic 'semantic' model for tests: it projects text onto four CONCEPT axes by
    synonym membership, so phrases that MEAN the same thing land near each other even with zero
    shared tokens (e.g. 'Moscow'/'sanctioned' vs 'Kremlin-linked'/'oligarch')."""
    semantic = True
    name = "fake-semantic:test"
    dim = 4
    CONCEPTS = [
        # 0: russia / sanctions
        {"russia", "russian", "moscow", "kremlin", "oligarch", "petrov", "oleg", "sanction",
         "sanctioned", "sanctions", "ofac", "blocked", "screen", "screened", "investigated",
         "watchlist", "designated"},
        # 1: credit / financials
        {"revenue", "ratio", "ratios", "credit", "underwrite", "underwriting", "ebitda",
         "financial", "financials", "liquidity", "leverage", "solvency"},
        # 2: adverse media
        {"adverse", "media", "fraud", "scandal", "laundering", "corruption", "bribery"},
        # 3: contact
        {"email", "phone", "contact", "linkedin", "outreach"},
    ]

    def embed_one(self, text):
        import re
        toks = set(re.findall(r"[a-z]+", (text or "").lower()))
        return _l2([float(len(toks & c)) for c in self.CONCEPTS])

    def embed_many(self, texts):
        return [self.embed_one(t) for t in texts]


# memory text and a query that share NO tokens/trigrams but mean the same thing
_MEMO = "Screened Oleg Petrov, Moscow businessman, against OFAC watchlist."
_QUERY = "Kremlin oligarch investigated, sanctions designated"


@pytest.fixture(autouse=True)
def _clean():
    reset_memory_for_tests()
    yield
    reset_memory_for_tests()


# --------------------------------------------------------------------------- #
# 1) A real embedding model is used when configured (not the lexical fallback)
# --------------------------------------------------------------------------- #
def test_real_embedder_selected_when_configured(monkeypatch):
    # default in tests is the lexical hashing fallback
    assert get_embedder().semantic is False
    # configure a hosted embedding model -> a real semantic embedder is selected
    monkeypatch.setenv("FINGENT_EMBED_BACKEND", "")
    monkeypatch.setenv("FINGENT_EMBED_API_KEY", "sk-test-key")
    reset_memory_for_tests()
    emb = get_embedder()
    assert isinstance(emb, OpenAICompatibleEmbedder)
    assert emb.semantic is True and emb.name.startswith("openai:")


# --------------------------------------------------------------------------- #
# 2) Semantic recall succeeds where lexical recall fails (the core fix)
# --------------------------------------------------------------------------- #
def test_semantic_recall_beats_keyword_overlap():
    # lexical (token-overlap) embedder: on a synonym query with no shared words, similarity is
    # near-noise (only incidental character-trigram collisions).
    set_embedder(HashingEmbedder())
    lex = LocalVectorMemory()
    lex.add("acme", "aml", _MEMO)
    lex_hits = lex.recall("acme", "aml", _QUERY, k=1)
    lex_score = lex_hits[0]["score"] if lex_hits else 0.0

    # semantic embedder: the SAME synonym query retrieves the memory by MEANING
    set_embedder(FakeSemanticEmbedder())
    sem = LocalVectorMemory()
    sem.add("acme", "aml", _MEMO)
    sem_hits = sem.recall("acme", "aml", _QUERY, k=1)
    assert sem_hits, "semantic recall should match meaning despite no shared words"
    assert sem_hits[0]["text"] == _MEMO
    sem_score = sem_hits[0]["score"]

    # the contrast is the point: semantic recall is a strong match where lexical is near-noise
    assert lex_score < 0.15, f"lexical score should be near-noise, got {lex_score}"
    assert sem_score > 0.6, f"semantic score should be strong, got {sem_score}"
    assert sem_score > lex_score * 3


# --------------------------------------------------------------------------- #
# 3) Durable across a 'restart': memory persists in the store, not in-process
# --------------------------------------------------------------------------- #
def test_memory_is_durable_across_restart(tmp_path):
    db = str(tmp_path / "mem.db")
    set_embedder(HashingEmbedder())             # deterministic, offline

    store1 = Store(db)
    SqlVectorMemory(store1).add("acme", "aml", "Oleg Petrov was screened for OFAC sanctions.")

    # simulate a process restart: a brand-new Store over the same database file
    store2 = Store(db)
    hits = SqlVectorMemory(store2).recall("acme", "aml", "Oleg Petrov sanctions", k=1)
    assert hits and "Oleg Petrov" in hits[0]["text"], "memory did not survive a restart"


# --------------------------------------------------------------------------- #
# 4) Tenant isolation is enforced in SQL, not by a Python dict key
# --------------------------------------------------------------------------- #
def test_sql_memory_tenant_isolation(tmp_path):
    set_embedder(HashingEmbedder())
    store = Store(str(tmp_path / "iso.db"))
    mem = SqlVectorMemory(store)
    mem.add("tenant_a", "aml", "Confidential: Oleg Petrov flagged on the OFAC list.")
    # the SAME query under a different tenant returns nothing
    assert mem.recall("tenant_a", "aml", "Oleg Petrov OFAC", k=3)
    assert mem.recall("tenant_b", "aml", "Oleg Petrov OFAC", k=3) == []


# --------------------------------------------------------------------------- #
# 5) Vectors from a different embedder are never mixed into recall
# --------------------------------------------------------------------------- #
def test_recall_does_not_mix_embedders(tmp_path):
    store = Store(str(tmp_path / "mix.db"))
    mem = SqlVectorMemory(store)
    set_embedder(FakeSemanticEmbedder())                 # store with model A (4-dim)
    mem.add("acme", "aml", _MEMO)
    set_embedder(HashingEmbedder())                      # recall with model B (512-dim)
    assert mem.recall("acme", "aml", _QUERY, k=3) == []  # mismatched model -> no garbage match


# --------------------------------------------------------------------------- #
# 6) Real hosted-embedding recall quality — runs only when a key is configured
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (os.getenv("FINGENT_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="no embedding API key configured; semantic recall eval requires a real model")
def test_real_embedding_recall_quality():
    set_embedder(OpenAICompatibleEmbedder())
    mem = LocalVectorMemory()
    mem.add("acme", "aml", "We screened Oleg Petrov, a sanctioned Russian businessman.")
    mem.add("acme", "aml", "Acme Corp Q3 revenue was $62M with healthy EBITDA margins.")
    mem.add("acme", "aml", "Resolved the CFO's email and phone for outreach.")
    hits = mem.recall("acme", "aml", "the Russian oligarch we investigated for sanctions", k=1)
    assert hits and "Petrov" in hits[0]["text"], hits
