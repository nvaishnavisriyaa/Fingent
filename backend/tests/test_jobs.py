"""
The async queue is DURABLE and recoverable, not a fire-and-forget thread pool.
  * a submitted run is a persisted job; draining runs it to a terminal job state
  * transient failures RETRY with backoff, then DEAD-LETTER after max_attempts
  * a run orphaned by a dead worker (expired lease) is RECOVERED, not stuck 'queued' forever
  * idempotency-key dedupes re-submits; jobs can be CANCELLED; claims are atomic
Workers are disabled in tests (FINGENT_JOB_WORKERS=0); the queue is driven via drain_once().
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest


def _agent(fp):
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers={"name": "aml", "lists": ["OFAC"]}, tenant_id="acme"))


@pytest.fixture
def fp():
    f = Fingent()
    _agent(f)
    return f


def test_submit_persists_a_job_and_drains_to_terminal(fp):
    run_id = fp.jobs.submit("acme", "aml", {"name": "Jane"})
    job = fp.store.get_job("acme", run_id)
    assert job and job["status"] == "queued"                  # durable, not just an in-memory future
    assert fp.store.get_run("acme", run_id)["status"] == "queued"
    assert fp.jobs.drain_once() is True                       # a worker claims + runs it
    assert fp.store.get_job("acme", run_id)["status"] == "succeeded"
    assert fp.store.get_run("acme", run_id)["status"] in ("success", "needs_review", "blocked")
    assert fp.jobs.drain_once() is False                      # queue now idle


def test_transient_failure_retries_then_dead_letters(fp, monkeypatch):
    fp.jobs.max_attempts = 2
    calls = {"n": 0}
    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("LLM timeout")
    monkeypatch.setattr(fp, "run_task", boom)
    run_id = fp.jobs.submit("acme", "aml", {"name": "Jane"})

    fp.jobs.drain_once()                                       # attempt 1 -> fail -> requeue (backoff)
    j = fp.store.get_job("acme", run_id)
    assert j["status"] == "queued" and j["attempts"] == 1 and "timeout" in (j["last_error"] or "")
    fp.store.db.execute("UPDATE jobs SET available_at=0 WHERE id=?", (run_id,))   # skip the backoff wait
    fp.store.db.commit()

    fp.jobs.drain_once()                                       # attempt 2 -> fail -> DEAD-LETTER
    j = fp.store.get_job("acme", run_id)
    assert j["status"] == "dead" and j["attempts"] == 2
    assert fp.store.get_run("acme", run_id)["status"] == "failed"
    assert calls["n"] == 2


def test_orphaned_running_job_is_recovered(fp):
    run_id = fp.jobs.submit("acme", "aml", {"name": "Jane"})
    job = fp.store.claim_job("workerA", lease_seconds=300)     # a worker claims it...
    assert job["id"] == run_id and job["status"] == "running"
    # ...then that worker/process dies: force its lease to expire
    fp.store.db.execute("UPDATE jobs SET lease_until=? WHERE id=?", (time.time() - 1, run_id))
    fp.store.db.commit()
    assert fp.store.recover_orphans() == 1                     # startup recovery requeues it
    assert fp.store.get_job("acme", run_id)["status"] == "queued"
    # and it can now be claimed + run again (not stuck forever)
    assert fp.jobs.drain_once() is True
    assert fp.store.get_job("acme", run_id)["status"] == "succeeded"


def test_atomic_claim_never_double_runs(fp):
    fp.jobs.submit("acme", "aml", {"name": "Jane"})
    first = fp.store.claim_job("w1", 300)
    second = fp.store.claim_job("w2", 300)                     # second worker gets nothing
    assert first is not None and second is None


def test_idempotency_key_dedupes_resubmits(fp):
    a = fp.jobs.submit("acme", "aml", {"name": "Jane"}, idempotency_key="job-42")
    b = fp.jobs.submit("acme", "aml", {"name": "Jane"}, idempotency_key="job-42")
    assert a == b                                             # same run, not two
    assert len(fp.store.list_jobs("acme")) == 1


def test_cancel_a_queued_job(fp):
    run_id = fp.jobs.submit("acme", "aml", {"name": "Jane"})
    assert fp.cancel_run("acme", run_id)["ok"] is True
    assert fp.store.get_job("acme", run_id)["status"] == "cancelled"
    assert fp.store.get_run("acme", run_id)["status"] == "cancelled"
    assert fp.jobs.drain_once() is False                      # cancelled work is never claimed/run
