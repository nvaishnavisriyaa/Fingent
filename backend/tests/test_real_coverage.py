"""Every Fingent agent must be wired to at least one REAL external data source — no agent runs
on mock/pure-computation alone. This guards the 'real tools, real data' product promise: if a new
template is added without a live source, this test fails."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fingent.templates import CATALOG

# Tools that call a live EXTERNAL data source (free public APIs or keyed providers).
REAL_EXTERNAL_TOOLS = {
    "edgar_search", "news_monitor", "enrich_company", "find_persona", "resolve_contact",
    "ofac_screen", "adverse_media_search", "pep_check", "reg_feed_ingest", "web_search",
    "verify_entity", "company_financials", "bank_lookup", "fx_rate", "treasury_rates",
    "account_lookup",
}

# Pure orchestration / reducer / cross-cutting agents that operate on UPSTREAM real data rather
# than fetching their own — legitimately exempt from holding their own external source.
EXEMPT = {"planner", "synthesis", "guardrail_compliance_overseer"}


def test_every_template_grants_a_real_external_source():
    for tpl in CATALOG:
        if tpl.name in EXEMPT:
            continue
        grantable = set(tpl.grantable_tools or [])
        assert grantable & REAL_EXTERNAL_TOOLS, (
            f"template '{tpl.name}' has no real external data source in {grantable}")


def test_default_grants_include_a_real_external_source():
    """The DEFAULT (least-privilege) grant of each non-exempt template must already include a real
    external source, so a freshly-created agent works on real data without extra configuration."""
    for tpl in CATALOG:
        if tpl.name in EXEMPT:
            continue
        defaults = set(tpl.fixed.get("required_tools", []))
        assert defaults & REAL_EXTERNAL_TOOLS, (
            f"template '{tpl.name}' default grant {defaults} has no real external source")
