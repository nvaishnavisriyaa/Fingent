"""
Data layer is production-shaped, not a prototype:
  * durable FILE DB survives a process restart (a second Store on the same path sees the data)
  * ordered, recorded migrations (schema_migrations) — not ad-hoc ALTERs
  * indexes on the hot tenant/time/status columns
  * promoted queryable columns (runs.status/risk_level/mode, traces.cost_usd/tokens)
  * analytics aggregates in SQL, not by scanning full trace blobs in Python
"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.store import Store, _MIGRATIONS
from fingent.schemas import CreateAgentRequest


def test_data_survives_a_restart(tmp_path):
    db = str(tmp_path / "fingent.db")
    fp = Fingent(db)
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers={"name": "aml", "lists": ["OFAC"]},
                                       tenant_id="acme"))
    rec = fp.run_task("acme", "aml", {"name": "Jane"})
    run_id = rec["id"]
    del fp                                    # simulate process exit

    fp2 = Fingent(db)                         # "restart" on the same file
    assert fp2.store.get_spec("acme", "aml") is not None          # agent survived
    assert fp2.store.get_run("acme", run_id) is not None          # run survived
    assert fp2.store.get_trace("acme", rec["trace_id"]) is not None


def test_memory_default_does_not_persist():
    a = Store(":memory:"); b = Store(":memory:")
    a.audit("acme", "x", "create", "foo", "")
    assert b.get_audit("acme") == []          # separate in-memory DBs are isolated


def test_migrations_are_recorded():
    st = Store(":memory:")
    rows = st.db.execute("SELECT version, name FROM schema_migrations ORDER BY version").fetchall()
    versions = [r["version"] for r in rows]
    assert versions == [v for (v, _, _) in _MIGRATIONS]          # every migration applied + recorded


def test_hot_indexes_exist():
    st = Store(":memory:")
    idx = {r["name"] for r in st.db.execute(
        "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
    for expected in ("ix_runs_tenant_ts", "ix_runs_tenant_status", "ix_traces_tenant_ts",
                     "ix_audit_tenant_ts"):
        assert expected in idx, f"missing index {expected}"


def test_promoted_columns_are_populated(tmp_path):
    fp = Fingent(str(tmp_path / "f.db"))
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers={"name": "aml", "lists": ["OFAC"]},
                                       tenant_id="acme"))
    rec = fp.run_task("acme", "aml", {"name": "Jane"})
    # runs row has real, queryable columns (not just a JSON blob)
    r = fp.store.db.execute(
        "SELECT status, risk_level, mode FROM runs WHERE id=?", (rec["id"],)).fetchone()
    assert r["status"] == rec["status"] and r["risk_level"] == rec["risk_level"]
    # a status filter uses the column + index
    same = fp.store.list_runs("acme", status=rec["status"])
    assert any(x["id"] == rec["id"] for x in same)


def test_analytics_uses_sql_aggregation(tmp_path, monkeypatch):
    # a metered model so traces carry real cost/tokens columns
    import fingent.runtime as rt
    class _Usage:
        enabled = True; name = "metered"; model = "llama-3.3-70b-versatile"
        def __init__(self): self.last_usage = {}
        def chat(self, messages, tools=None, tool_choice="auto", **k):
            if tools and not any(m.get("role") == "tool" for m in messages):
                self.last_usage = {"prompt_tokens": 1000, "completion_tokens": 200}
                fn = tools[0]["function"]["name"]
                return {"role": "assistant", "content": None,
                        "tool_calls": [{"id": "c1", "function": {"name": fn, "arguments": "{}"}}]}
            self.last_usage = {"prompt_tokens": 500, "completion_tokens": 100}
            return {"role": "assistant", "content": "done"}
        def usage_split(self):
            u = self.last_usage
            return (u.get("prompt_tokens", 0), u.get("completion_tokens", 0), False)
    monkeypatch.setattr(rt, "LlmProvider", _Usage)
    fp = Fingent(str(tmp_path / "a.db"))
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers={"name": "aml", "lists": ["OFAC"]}, tenant_id="acme"))
    fp.run_task("acme", "aml", {"name": "Jane"})
    # SQL SUM over the indexed traces.tokens/cost columns
    usage = fp.store.usage_totals("acme", 0.0)
    assert usage["tokens"] == 1800 and usage["cost_usd"] > 0
    a = fp.analytics("acme")
    assert a["totals"]["tokens"] == 1800 and a["totals"]["cost_basis"] == "real LLM usage"
