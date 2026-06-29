"""PII is never persisted in cleartext. A KYC agent must execute on the REAL identifiers (so it
can verify them) but the stored RunRecord (input + steps) must be redacted — a compliance dump of
the runs table must not contain a raw SSN, card or email."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from fingent import Fingent
from fingent.schemas import CreateAgentRequest

SSN = "123-45-6789"
EMAIL = "john.doe@example.com"


def test_run_input_and_steps_are_redacted_at_rest():
    fp = Fingent()
    fp.create_agent(CreateAgentRequest(template="kyc_onboarding", answers={"name": "kyc"},
                                       tenant_id="acme"))
    fp.deploy_agent("acme", "kyc", actor="t")
    rec = fp.run_task("acme", "kyc",
                      {"name": "John Doe", "id_number": SSN, "ssn": SSN, "email": EMAIL})
    blob = json.dumps(rec)
    assert SSN not in blob, "raw SSN must not be stored in the run record"
    assert EMAIL not in blob, "raw email must not be stored in the run record"
    assert "[REDACTED_SSN]" in blob or "[REDACTED_EMAIL]" in blob, "redaction markers expected"

    # and the stored run fetched back from the store is also clean
    stored = json.dumps(fp.store.get_run("acme", rec["id"]))
    assert SSN not in stored and EMAIL not in stored


def test_contact_agent_shows_email_but_does_not_store_it():
    """A contact agent (pii_allow=email/phone) RETURNS the email to the caller, but the persisted
    RunRecord must still be redacted — display and storage are decoupled."""
    fp = Fingent()
    fp.create_agent(CreateAgentRequest(template="contact", answers={"name": "ct"},
                                       tenant_id="acme"), auto_provision=True)
    fp.deploy_agent("acme", "ct", actor="t")
    rec = fp.run_task("acme", "ct", {"name": "John Doe", "company": "Acme"})
    returned = json.dumps(rec)
    stored = json.dumps(fp.store.get_run("acme", rec["id"]))
    # the caller SEES the resolved email (pii_allow honored in the response)
    assert "john.doe@acme.com" in returned
    # but the database does NOT retain the raw email (always redacted at rest)
    assert "john.doe@acme.com" not in stored
    assert "[REDACTED_EMAIL]" in stored
