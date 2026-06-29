"""
Tool integrity tests — the live tool paths must return REAL data, never fabricated values,
and must never contact a fake/placeholder endpoint.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import fingent.tools_native as t


class _Resp:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok
        self.status_code = 200 if ok else 500

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("http error")


def test_enrich_company_does_not_fabricate_firmographics(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("ENRICH_API_URL", raising=False)
    # simulate exactly what Clearbit autocomplete actually returns: name/domain/logo only
    monkeypatch.setattr(t, "_get",
                        lambda *a, **k: _Resp([{"name": "Stripe", "domain": "stripe.com",
                                                "logo": None}]))
    r = t.enrich_company("Stripe")
    assert r["source"] == "live:Clearbit autocomplete"
    assert r["company"] == "Stripe" and r["domain"] == "stripe.com"
    # the previously hardcoded fabrications must be gone
    assert r["employees"] is None and r["revenue_est_usd"] is None
    assert r["employees"] != 480 and r["revenue_est_usd"] != 62_000_000
    assert r["hq"] is None and r["industry"] is None


def test_enrich_company_uses_real_provider_when_configured(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.setenv("ENRICH_API_URL", "https://enrich.internal/company")
    seen = {}

    def fake_get(url, params=None, **k):
        seen["url"] = url
        return _Resp({"employees": 8000, "revenue_est_usd": 1_400_000_000, "hq": "South SF"})

    monkeypatch.setattr(t, "_get", fake_get)
    r = t.enrich_company("Stripe")
    assert seen["url"] == "https://enrich.internal/company"
    assert r["source"] == "live:enrich_api" and r["employees"] == 8000


def test_identity_verify_never_calls_a_fake_endpoint(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("KYC_API_URL", raising=False)
    monkeypatch.delenv("KYC_API_KEY", raising=False)

    def boom(*a, **k):
        raise AssertionError("identity_verify must not make a network call without KYC_API_URL")

    monkeypatch.setattr(t, "_post", boom)
    r = t.identity_verify("Jane Smith", "AB-123456")
    assert r["source"] == "computed" and r["verified"] is True


def test_identity_verify_posts_to_configured_url(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.setenv("KYC_API_URL", "https://kyc.internal/verify")
    seen = {}

    def fake_post(url, json=None, headers=None, **k):
        seen["url"] = url
        return _Resp({"verified": True, "confidence": 0.97})

    monkeypatch.setattr(t, "_post", fake_post)
    r = t.identity_verify("Jane Smith", "AB-123456")
    assert seen["url"] == "https://kyc.internal/verify"
    assert r["source"] == "live:KYC provider" and r["confidence"] == 0.97


def test_ofac_prefers_current_sanctions_list_service():
    # Treasury's current host must be tried first, with the legacy URL as fallback
    assert t._OFAC_URLS[0].startswith("https://sanctionslistservice.ofac.treas.gov")
    assert any("treasury.gov" in u for u in t._OFAC_URLS)


# --------------------------------------------------------------------------- #
# Integrity: tools must NOT fabricate realistic data in live mode when their
# real source is unavailable (missing key / no match / no endpoint).
# --------------------------------------------------------------------------- #
def test_find_persona_does_not_fabricate_people_in_live_mode(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("PEOPLE_DATA_API_KEY", raising=False)
    r = t.find_persona("Acme Corp")
    # no fabricated PEOPLE...
    assert r["personas"] == []
    names = str(r)
    assert "Jane Doe" not in names and "Sam Lee" not in names
    # ...but the user gets the right TITLES to target (a real heuristic, clearly labelled)
    assert r["source"] == "computed:persona-heuristic"
    assert r["target_titles"]


def test_resolve_contact_does_not_fabricate_email_in_live_mode(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("HUNTER_API_KEY", raising=False)
    r = t.resolve_contact("Jane Doe", "acme.com")
    # a best-guess email is provided so the agent is useful, BUT it is clearly marked UNVERIFIED
    # (verified=False, source is 'computed', not 'live') — never presented as confirmed/real data.
    assert r["source"] == "computed:email-heuristic"
    assert r["verified"] is False and r["live"] is False
    assert r["email"] == "jane.doe@acme.com" and r["candidates"]
    assert r.get("status") == "unverified_guess" and "unverified" in r["note"].lower()


def test_resolve_contact_derives_domain_from_company_name(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("HUNTER_API_KEY", raising=False)
    r = t.resolve_contact("Patrick Collison", "Stripe")   # company NAME, not a domain
    assert r["company_domain"] == "stripe.com"
    assert r["email"] == "patrick.collison@stripe.com" and r["verified"] is False


def test_account_lookup_does_not_fabricate_balance_in_live_mode(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("ACCOUNT_API_URL", raising=False)
    r = t.account_lookup("ACC-77")
    assert r["source"] == "unavailable" and r["live"] is False
    assert "balance_usd" not in r


def test_ocr_extract_does_not_fabricate_statement_in_live_mode(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("OCR_API_URL", raising=False)
    r = t.ocr_extract("statement.pdf")
    assert r["source"] == "unavailable" and r["live"] is False
    assert r["text"] == "" and "1,204,332" not in str(r)


def test_ocr_extract_uses_configured_provider(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.setenv("OCR_API_URL", "https://ocr.internal/extract")
    seen = {}

    def fake_post(url, json=None, **k):
        seen["url"] = url
        return _Resp({"text": "REAL TEXT", "fields": {"closing_balance": 999}})

    monkeypatch.setattr(t, "_post", fake_post)
    r = t.ocr_extract("statement.pdf")
    assert seen["url"] == "https://ocr.internal/extract"
    assert r["source"] == "live:OCR provider" and r["fields"]["closing_balance"] == 999


def test_offline_mode_still_returns_sample_data(monkeypatch):
    # demo/offline (FINGENT_LIVE_DATA=0) keeps the sample fallbacks for a runnable demo
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    # these no longer fabricate named people / contacts offline — they return REAL heuristics
    assert t.find_persona("Acme")["source"] == "computed:persona-heuristic"
    assert t.resolve_contact("Jane", "acme.com")["source"] == "computed:email-heuristic"
    assert t.account_lookup("ACC-1")["source"] == "mock"
    assert t.ocr_extract("x.pdf")["source"] == "mock"


# --------------------------------------------------------------------------- #
# Comprehensive: NO tool fabricates sample data in live mode. document_intelligence
# regression (parse_financials returned 62,000,000 mock) + the rest of the catalog.
# --------------------------------------------------------------------------- #
import pytest as _pytest


@_pytest.mark.parametrize("call", [
    lambda: t.parse_financials(""),
    lambda: t.risk_score(),
    lambda: t.anomaly_detect([100]),
    lambda: t.enrich_company(""),
    lambda: t.news_monitor(""),
    lambda: t.edgar_search(""),
    lambda: t.reg_feed_ingest("US"),
    lambda: t.ofac_screen(""),
    lambda: t.pep_check(""),
    lambda: t.adverse_media_search(""),
    lambda: t.web_search(""),
])
def test_no_fabrication_in_live_mode(monkeypatch, call):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    # force live data sources to be 'unreachable' so we exercise the fallback path
    monkeypatch.setattr(t, "_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(t, "_post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(t, "_rss_titles", lambda *a, **k: [])
    r = call()
    assert r.get("source") != "mock", f"tool fabricated mock data in live mode: {r}"
    # the hardcoded demo financials must never appear
    assert 62_000_000 not in r.values()


def test_parse_financials_still_parses_real_text(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    r = t.parse_financials("Revenue 5,000,000 EBITDA 1,200,000 total debt 800,000 "
                           "current assets 2,000,000 current liabilities 1,000,000")
    assert r["source"] == "live:parsed" and r["revenue"] == 5_000_000


def test_offline_mode_keeps_labeled_samples(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    assert t.parse_financials("")["source"] == "mock"   # demo mode keeps sample, clearly labeled
