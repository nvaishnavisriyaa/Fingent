"""
Durable background job queue — runs agents OFF the request thread, and survives a restart.

Unlike a bare ThreadPool, the queue STATE lives in the store (a `jobs` state machine), so:
  * a run enqueued before a crash is NOT orphaned — on restart `recover_orphans()` requeues any
    job whose worker lease expired, and pollers pick it up again;
  * claims are ATOMIC (one UPDATE), so multiple worker threads — and multiple PROCESSES sharing
    the same DB/Postgres — never run the same job twice;
  * transient failures RETRY with exponential backoff, then DEAD-LETTER after `max_attempts`;
  * an `idempotency_key` dedupes re-submits;
  * jobs can be CANCELLED.

The in-process pool here is the default worker. Because all coordination is in the store, you can
also run this exact worker loop in a separate process/container against the same Postgres — or swap
the loop for Arq/Celery/RQ — without changing the platform or runtime. State machine:

    queued --claim--> running --ok--> succeeded
                         |--fail & attempts<max--> queued (backoff)
                         |--fail & attempts>=max--> dead
    queued|running --cancel--> cancelled
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid


class JobRunner:
    def __init__(self, platform, workers: int | None = None) -> None:
        self.platform = platform
        self.workers = workers if workers is not None else int(os.getenv("FINGENT_JOB_WORKERS", "4"))
        self.lease = float(os.getenv("FINGENT_JOB_LEASE", "300"))          # seconds a claim is held
        self.max_attempts = int(os.getenv("FINGENT_JOB_MAX_ATTEMPTS", "3"))
        self.poll_idle = float(os.getenv("FINGENT_JOB_POLL_SECONDS", "0.2"))
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}"
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._started = False
        self._lock = threading.Lock()

    # ----- enqueue ------------------------------------------------------- #
    def submit(self, tenant_id: str, agent: str, inputs: dict,
               approve_side_effecting: bool = False, idempotency_key: str | None = None) -> str:
        store = self.platform.store
        if idempotency_key:                                   # dedupe re-submits
            existing = store.find_job_by_key(tenant_id, idempotency_key)
            if existing:
                return existing["id"]
        run_id = "run_" + uuid.uuid4().hex[:12]
        store.save_run({
            "id": run_id, "tenant_id": tenant_id, "agent": agent, "trace_id": "",
            "mode": "", "input": inputs or {}, "status": "queued", "steps": [],
            "output": None, "risk_score": 0, "risk_level": "low", "risk_flags": [],
            "pending_action": None, "duration_ms": 0.0, "ts": time.time()})
        store.enqueue_job(run_id, tenant_id, agent,
                          json.dumps({"inputs": inputs or {},
                                      "approve_side_effecting": approve_side_effecting}),
                          max_attempts=self.max_attempts, idempotency_key=idempotency_key)
        store.audit(tenant_id, agent, "enqueue", run_id, {"status": "queued"})
        self.start()                                          # ensure workers are running
        return run_id

    def cancel(self, tenant_id: str, run_id: str) -> bool:
        ok = self.platform.store.cancel_job(tenant_id, run_id)
        if ok:
            rec = self.platform.store.get_run(tenant_id, run_id)
            if rec and rec.get("status") in ("queued", "running"):
                rec["status"] = "cancelled"
                self.platform.store.save_run(rec)
            self.platform.store.audit(tenant_id, run_id, "cancel", run_id, {})
        return ok

    # ----- worker lifecycle --------------------------------------------- #
    def start(self) -> None:
        """Start the poller threads once. Recovers orphaned runs from a previous process first."""
        if self.workers <= 0:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
            try:
                recovered = self.platform.store.recover_orphans()
                if recovered:
                    self.platform.store.audit("*", "jobrunner", "recover_orphans", "-",
                                              {"recovered": recovered})
            except Exception:  # noqa: BLE001 — recovery must never block startup
                pass
            for i in range(self.workers):
                t = threading.Thread(target=self._poll_loop, name=f"fingent-worker-{i}",
                                     daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                did = self.drain_once()
            except Exception:  # noqa: BLE001 — a poller must never die silently
                did = False
            if not did:
                self._stop.wait(self.poll_idle)

    # ----- one unit of work (also the synchronous, testable entry point) - #
    def drain_once(self) -> bool:
        """Claim and run one job. Returns True if a job was processed, False if the queue is idle.
        Called by the poller loop; also callable directly for deterministic/synchronous draining."""
        job = self.platform.store.claim_job(self.worker_id, self.lease)
        if not job:
            return False
        self._run_job(job)
        return True

    def _run_job(self, job: dict) -> None:
        store = self.platform.store
        run_id, tenant, agent = job["id"], job["tenant_id"], job["agent"]
        payload = {}
        try:
            payload = json.loads(job.get("payload") or "{}")
        except Exception:  # noqa: BLE001
            payload = {}

        # honor a cancellation that landed between claim and run
        cur = store.get_job(tenant, run_id)
        if cur and cur["status"] == "cancelled":
            return

        rec = store.get_run(tenant, run_id)
        if rec:
            rec["status"] = "running"
            store.save_run(rec)

        try:
            self.platform.run_task(
                tenant, agent, payload.get("inputs") or {},
                approve_side_effecting=payload.get("approve_side_effecting", False),
                run_id=run_id)
            store.finish_job(run_id, "succeeded")
        except Exception as e:  # noqa: BLE001 — durable failure handling: retry then dead-letter
            err = str(e)[:300]
            attempts = int(job.get("attempts", 1))
            max_a = int(job.get("max_attempts", self.max_attempts))
            if attempts < max_a:
                backoff = min(60.0, 2.0 ** attempts)          # exponential, capped
                store.requeue_job(run_id, time.time() + backoff, err)
                store.audit(tenant, agent, "run_retry", run_id,
                            {"attempt": attempts, "backoff_s": backoff, "error": err[:160]})
                rec = store.get_run(tenant, run_id)
                if rec:
                    rec["status"] = "queued"
                    rec["risk_flags"] = list(rec.get("risk_flags", [])) + [f"retry:{attempts}"]
                    store.save_run(rec)
            else:
                store.finish_job(run_id, "dead", err)         # dead-letter
                store.audit(tenant, agent, "run_dead_letter", run_id,
                            {"attempts": attempts, "error": err[:160]})
                rec = store.get_run(tenant, run_id) or {
                    "id": run_id, "tenant_id": tenant, "agent": agent, "trace_id": "",
                    "mode": "", "input": payload.get("inputs") or {}, "ts": time.time()}
                rec["status"] = "failed"
                rec["output"] = {"error": err, "attempts": attempts}
                store.save_run(rec)
