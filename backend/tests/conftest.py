"""Keep the acceptance tests deterministic and offline.

The native tools call live APIs by default (FINGENT_LIVE_DATA=1). For the test suite we force
the offline deterministic fallback so tests are fast and reproducible without network access.

We also neutralize any LLM credentials so the spec compiler exercises its offline heuristic
engine deterministically. These are set BEFORE `fingent` is imported, so the package's
load_dotenv() call (override=False) will not re-populate them from a developer's local .env.
"""
import os

os.environ["FINGENT_LIVE_DATA"] = "0"
# the deterministic engine is gated; tests opt in explicitly (never the product default)
os.environ["FINGENT_ALLOW_DEMO"] = "1"
# tests run on an isolated in-memory DB (the served app defaults to a durable file)
os.environ["FINGENT_DB"] = ":memory:"
# no background worker threads in tests — drive the queue deterministically via drain_once()
os.environ["FINGENT_JOB_WORKERS"] = "0"

# Force the offline compiler path regardless of any real keys in the developer's .env.
# Empty (not deleted) so load_dotenv(override=False) leaves them empty.
for _k in ("GROQ_API_KEY", "FINGENT_LLM_API_KEY", "GROQ_MODEL", "FINGENT_LLM_MODEL",
           "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
    os.environ[_k] = ""

# Deterministic, offline embeddings for memory tests: force the lexical hashing embedder and
# neutralize any embedding API keys so no test makes a network call. Recall-quality tests that
# need a real semantic model inject one explicitly via memory.set_embedder().
os.environ["FINGENT_EMBED_BACKEND"] = "hashing"
for _k in ("FINGENT_EMBED_API_KEY", "OPENAI_API_KEY"):
    os.environ[_k] = ""
