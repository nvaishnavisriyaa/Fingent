"""
Template catalog (§12).

Adding an agent type is a *config-only* change: append an AgentTemplate here (or crystallize one
from a from-scratch build, see compiler.py). No code change is ever needed to add an agent.
The HTML form is a pure rendering of `parameters` + the mandatory free-text field.
"""
from __future__ import annotations

from .schemas import (
    AgentTemplate, Dependency, DependencyType, GuardrailPolicy, TemplateParameter,
)

HARD = DependencyType.HARD
SOFT = DependencyType.SOFT


def _name_param() -> TemplateParameter:
    return TemplateParameter(name="name", type="text", label="Agent name", required=True)


CATALOG: list[AgentTemplate] = [
    # ------------------------------ TIER 1 (GTM) --------------------------- #
    AgentTemplate(
        name="planner", tier=1,
        description="Supervisor: reads ICP + persona config, emits an agent-call DAG.",
        fixed={"base_role": "You are the GTM planner. Read ICP/persona config and emit a "
                            "structured plan (DAG) of agent calls to discover and qualify FS "
                            "prospects."},
        parameters=[_name_param(),
                    TemplateParameter(name="icp", type="text", label="Ideal customer profile",
                                      default="US fintechs, 100-1000 employees, raised debt"),
                    ],
        grantable_tools=["web_search"],
    ),
    AgentTemplate(
        name="signal_trigger", tier=1,
        description="Watch web/market for FS triggers (funding, new CFO/Treasurer, 8-K, expansion, debt).",
        fixed={"base_role": "Detect FS buying-trigger signals for target companies."},
        parameters=[_name_param(),
                    TemplateParameter(name="trigger_types", type="multi_select",
                                      label="Trigger types to watch",
                                      options=["funding", "new_cfo", "8-K", "expansion", "debt"],
                                      default=["funding", "new_cfo"]),
                    ],
        grantable_tools=["news_monitor", "edgar_search", "web_search"],
    ),
    AgentTemplate(
        name="icp_matching", tier=1,
        description="Score companies vs firmographic + financial-health ICP.",
        fixed={"base_role": "Score companies against the firmographic + financial-health ICP."},
        parameters=[_name_param(),
                    TemplateParameter(name="min_score", type="number", label="Minimum match score (0-1)",
                                      default=0.6, min=0, max=1),
                    ],
        default_depends_on=[Dependency(agent="signal_trigger", type=SOFT,
                                       reason="scores the companies that signals surfaced")],
        grantable_tools=["enrich_company", "web_search"],
    ),
    AgentTemplate(
        name="enrichment_validation", tier=1,
        description="Enrich/validate a matched company.",
        fixed={"base_role": "Enrich and validate matched companies with firmographic data."},
        parameters=[_name_param()],
        default_depends_on=[Dependency(agent="icp_matching", type=HARD,
                                       reason="needs matched companies to enrich")],
        grantable_tools=["enrich_company", "edgar_search", "web_search"],
    ),
    AgentTemplate(
        name="persona_decision_maker", tier=1,
        description="Find CFO/VP Finance/Treasurer/Controller/Head of Risk.",
        fixed={"base_role": "Identify financial decision-maker personas at the target company."},
        parameters=[_name_param(),
                    TemplateParameter(name="titles", type="multi_select", label="Target titles",
                                      options=["CFO", "VP Finance", "Treasurer", "Controller",
                                               "Head of Risk"],
                                      default=["CFO", "VP Finance"]),
                    ],
        default_depends_on=[Dependency(agent="enrichment_validation", type=HARD,
                                       reason="needs validated companies")],
        grantable_tools=["find_persona", "web_search"],
    ),
    AgentTemplate(
        name="contact", tier=1,
        description="Resolve email/phone/LinkedIn.",
        fixed={"base_role": "Resolve verified contact details for identified personas."},
        parameters=[_name_param()],
        default_depends_on=[Dependency(agent="persona_decision_maker", type=HARD,
                                       reason="needs personas to resolve contacts for")],
        grantable_tools=["resolve_contact", "web_search"],
    ),
    AgentTemplate(
        name="synthesis", tier=1,
        description="Actionable summary + next action.",
        fixed={"base_role": "Synthesize an actionable account summary and recommend the next action."},
        parameters=[_name_param()],
        default_depends_on=[
            Dependency(agent="enrichment_validation", type=HARD, reason="needs enriched company"),
            Dependency(agent="persona_decision_maker", type=HARD, reason="needs personas"),
            Dependency(agent="contact", type=HARD, reason="needs contact details"),
        ],
        grantable_tools=["web_search"],
    ),

    # --------------------------- TIER 2 (FS ops) -------------------------- #
    AgentTemplate(
        name="document_intelligence", tier=2,
        description="OCR + LLM: statements/tax/financials -> structured JSON (foundational).",
        fixed={"base_role": "Extract structured JSON from financial documents via OCR + parsing."},
        parameters=[_name_param(),
                    TemplateParameter(name="doc_types", type="multi_select",
                                      label="Document types to process",
                                      options=["bank_statements", "tax_returns",
                                               "financial_statements"],
                                      default=["bank_statements", "financial_statements"]),
                    ],
        grantable_tools=["ocr_extract", "parse_financials"],
    ),
    AgentTemplate(
        name="kyc_onboarding", tier=2,
        description="Identity verification, doc checks, watchlist hits.",
        fixed={"base_role": "Run KYC: verify identity, check documents, screen watchlists."},
        parameters=[_name_param()],
        default_depends_on=[Dependency(agent="document_intelligence", type=SOFT,
                                       reason="uses parsed ID documents when available")],
        grantable_tools=["ocr_extract", "ofac_screen", "pep_check", "web_search"],
    ),
    AgentTemplate(
        name="aml_sanctions_screening", tier=2,
        description="OFAC/EU/UN/PEP + adverse-media.",
        fixed={"base_role": "Screen names/entities against sanctions, PEP, and adverse media."},
        parameters=[_name_param(),
                    TemplateParameter(name="lists", type="multi_select", label="Watchlists",
                                      options=["OFAC", "EU", "UN", "PEP"],
                                      default=["OFAC", "EU", "UN", "PEP"]),
                    ],
        grantable_tools=["ofac_screen", "adverse_media_search", "pep_check", "web_search"],
    ),
    AgentTemplate(
        name="credit_underwriting", tier=2,
        description="Parse financials -> ratios -> risk score (LoanGuard).",
        fixed={"base_role": "Underwrite credit: parse financials, compute ratios, produce a "
                            "risk score and a recommendation."},
        parameters=[
            _name_param(),
            TemplateParameter(name="doc_types", type="multi_select",
                              label="Document types to underwrite",
                              options=["bank_statements", "tax_returns", "financial_statements"],
                              default=["financial_statements"]),
            TemplateParameter(name="risk_threshold", type="number",
                              label="Risk score threshold (0-1)", default=0.7, min=0, max=1),
            TemplateParameter(name="requires_human_review", type="boolean",
                              label="Require human review before final decision?", default=True),
        ],
        default_depends_on=[Dependency(agent="document_intelligence", type=HARD,
                                       reason="needs parsed financial documents")],
        default_guardrails=GuardrailPolicy(output_review_required=True),
        grantable_tools=["parse_financials", "compute_ratios", "web_search"],
    ),
    AgentTemplate(
        name="fraud_anomaly", tier=2,
        description="Rules + anomaly detection over transactions.",
        fixed={"base_role": "Detect fraud via rules + anomaly detection over transactions."},
        parameters=[_name_param()],
        grantable_tools=["anomaly_detect"],
    ),
    AgentTemplate(
        name="compliance_monitoring", tier=2,
        description="Regulatory feed -> obligations.",
        fixed={"base_role": "Ingest regulatory feeds and produce obligations to track."},
        parameters=[_name_param(),
                    TemplateParameter(name="jurisdiction", type="select", label="Jurisdiction",
                                      options=["US", "EU", "UK"], default="US")],
        grantable_tools=["reg_feed_ingest", "web_search"],
    ),
    AgentTemplate(
        name="servicing_support", tier=2,
        description="Account inquiries.",
        fixed={"base_role": "Answer account servicing inquiries."},
        parameters=[_name_param()],
        grantable_tools=["web_search"],
    ),
    AgentTemplate(
        name="guardrail_compliance_overseer", tier=2,
        description="Vets flagged outputs before HITL (cross-cutting).",
        fixed={"base_role": "Compliance overseer: vet outputs for regulatory red-flags, leaked "
                            "PII, and unsupported claims; block or annotate."},
        parameters=[_name_param()],
        default_guardrails=GuardrailPolicy(output_review_required=True),
        grantable_tools=["web_search"],
    ),
]


def load_catalog(store) -> None:
    """Idempotently load the built-in templates into the store."""
    for tpl in CATALOG:
        if store.get_template(tpl.name) is None:
            store.save_template(tpl)
