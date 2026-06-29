"""
Real analytical engines behind the financial-services tools — the depth an analyst actually
needs, not thin wrappers. Pure-Python and dependency-light so the LOGIC is real and unit-tested
offline; the live DATA (OFAC SDN, Google News) is fetched by tools_native and fed in here.

  * sanctions entity resolution  — normalize + alias + fuzzy score + match classification
  * adverse-media NLP            — lexicon/category risk classification over real headlines
  * document extraction (OCR)    — real text from PDFs (pdfplumber/pypdf) and images (tesseract)
"""
from __future__ import annotations

import os
import re

# --------------------------------------------------------------------------- #
# 1) Sanctions entity resolution
# --------------------------------------------------------------------------- #
_TITLES = {"mr", "mrs", "ms", "dr", "sir", "the", "hon", "mr.", "dr."}
_CORP = {"llc", "ltd", "inc", "co", "corp", "corporation", "company", "plc", "gmbh",
         "sa", "ag", "bv", "oao", "ooo", "pjsc", "ojsc"}


def normalize_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    toks = [t for t in s.split() if t and t not in _TITLES and t not in _CORP]
    return " ".join(toks)


def _tokens(name: str) -> set:
    return set(normalize_name(name).split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_ratio(a: str, b: str) -> float:
    a, b = normalize_name(a), normalize_name(b)
    if not a and not b:
        return 1.0
    m = max(len(a), len(b)) or 1
    return 1.0 - levenshtein(a, b) / m


def token_set_ratio(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    return len(inter) / len(ta | tb)


def match_score(query: str, candidate: str) -> float:
    """Order-independent similarity in [0,1] combining token overlap and edit distance, with a
    boost when one name's tokens are fully contained in the other (e.g. middle name dropped)."""
    ts = token_set_ratio(query, candidate)
    lv = lev_ratio(query, candidate)
    ta, tb = _tokens(query), _tokens(candidate)
    subset = 1.0 if ta and tb and (ta <= tb or tb <= ta) else 0.0
    # per-token best edit match (handles transliteration: "Oleg"/"Olleg")
    tok_lv = 0.0
    if ta and tb:
        tok_lv = sum(max(lev_ratio(x, y) for y in tb) for x in ta) / len(ta)
    return round(max(ts, lv, 0.6 * ts + 0.4 * tok_lv, 0.9 * subset * tok_lv), 4)


def classify_match(score: float) -> str:
    if score >= 0.95:
        return "exact"
    if score >= 0.85:
        return "strong"
    if score >= 0.70:
        return "partial"
    return "weak"


def screen_names(query: str, names: list[str], threshold: float = 0.85,
                 limit: int = 8) -> dict:
    """Real sanctions screening: score the query against every SDN name, rank, classify, and
    decide a hit by threshold. Returns scored candidates an analyst can adjudicate."""
    q = (query or "").strip()
    scored = []
    for n in names:
        s = match_score(q, n)
        if s >= 0.70:
            scored.append({"name": n, "score": s, "match": classify_match(s)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:limit]
    best = top[0]["score"] if top else 0.0
    return {
        "query": q,
        "ofac_hit": best >= threshold,
        "best_score": best,
        "match_type": classify_match(best) if top else "none",
        "candidates": top,
        "screened_against": len(names),
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# 2) Adverse-media NLP
# --------------------------------------------------------------------------- #
# Risk lexicon by category with per-category severity weight (0-1).
_ADVERSE = {
    "financial_crime": (0.95, ["fraud", "embezzle", "money launder", "laundering", "ponzi",
                               "scam", "misappropriat", "wire fraud", "racketeer", "rico"]),
    "sanctions": (1.0, ["sanction", "ofac", "designated", "blocked person", "sdn",
                        "export control", "evading sanctions"]),
    "corruption": (0.9, ["bribe", "bribery", "kickback", "corruption", "graft", "fcpa"]),
    "terrorism": (1.0, ["terror", "terrorist financing", "financing of terrorism"]),
    "legal_action": (0.7, ["indict", "charged", "convict", "guilty", "lawsuit", "sued",
                           "settlement", "fine", "penalty", "fined", "prosecut", "plea"]),
    "regulatory": (0.6, ["investigation", "probe", "sec charges", "regulator", "subpoena",
                         "enforcement action", "cease and desist"]),
    "default_negative": (0.4, ["arrest", "raid", "seized", "collapse", "insolvenc",
                               "bankrupt", "default", "misconduct", "whistleblow"]),
}
_NEGATIONS = ("cleared of", "acquitted", "dropped", "dismissed", "not guilty", "exonerat")


def classify_headline(title: str) -> dict:
    t = (title or "").lower()
    negated = any(n in t for n in _NEGATIONS)
    cats, weight = [], 0.0
    for cat, (sev, terms) in _ADVERSE.items():
        if any(term in t for term in terms):
            cats.append(cat)
            weight = max(weight, sev)
    if negated:
        weight *= 0.25
    return {"headline": title, "categories": cats, "severity": round(weight, 3),
            "negated": negated}


def score_adverse(name: str, headlines: list[str]) -> dict:
    """Real adverse-media analysis: classify each headline into risk categories and produce an
    aggregate risk score (0-100) — not a raw count."""
    classified = [classify_headline(h) for h in headlines]
    hits = [c for c in classified if c["categories"]]
    cat_counts: dict = {}
    for c in hits:
        for cat in c["categories"]:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    # aggregate: severity-weighted, saturating
    sev = sum(c["severity"] for c in hits)
    risk = round(min(100.0, 100 * (1 - 2.718 ** (-0.6 * sev))), 1) if sev else 0.0
    band = "high" if risk >= 60 else ("medium" if risk >= 25 else ("low" if risk > 0 else "none"))
    return {
        "name": name,
        "screened": len(headlines),
        "adverse_hits": len(hits),
        "risk_score": risk,
        "risk_band": band,
        "categories": cat_counts,
        "flagged": [{"headline": c["headline"], "categories": c["categories"],
                     "severity": c["severity"]} for c in hits][:8],
    }


# --------------------------------------------------------------------------- #
# 3) Real document extraction (OCR + parsing)
# --------------------------------------------------------------------------- #
_AMOUNT = re.compile(r"(?:USD|EUR|GBP|\$|€|£)\s?([\d,]+(?:\.\d{2})?)|\b([\d,]{4,}(?:\.\d{2})?)\b")
_DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|"
                   r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
                   re.I)
_ACCT = re.compile(r"\b(?:a/c|acct|account)\s*(?:no\.?|number|#)?\s*[:#]?\s*([A-Z0-9-]{6,})\b", re.I)
_IBAN = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{10,30})\b")


def _extract_fields(text: str) -> dict:
    amts = []
    for m in _AMOUNT.finditer(text):
        raw = m.group(1) or m.group(2)
        try:
            amts.append(float(raw.replace(",", "")))
        except (ValueError, AttributeError):
            pass
    fields: dict = {}
    if amts:
        fields["amounts_found"] = len(amts)
        fields["max_amount"] = max(amts)
        # closing/ending balance heuristic
        mb = re.search(r"(?:closing|ending|available)\s+balance[^\d]{0,15}"
                       r"(?:USD|EUR|GBP|\$|€|£)?\s?([\d,]+(?:\.\d{2})?)", text, re.I)
        if mb:
            fields["closing_balance"] = float(mb.group(1).replace(",", ""))
    dates = _DATE.findall(text)
    if dates:
        fields["dates_found"] = [d if isinstance(d, str) else d[0] for d in dates][:6]
    acct = _ACCT.search(text) or _IBAN.search(text)
    if acct:
        fields["account_number"] = acct.group(1)
    return fields


def extract_document(path_or_bytes, filename: str = "") -> dict:
    """Real extraction: PDF text via pdfplumber/pypdf, images via Tesseract OCR. Returns
    {text, fields, pages, method} or raises/returns empty text if nothing is readable."""
    import io
    data = None
    name = filename
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    elif isinstance(path_or_bytes, str) and os.path.isfile(path_or_bytes):
        name = path_or_bytes
        with open(path_or_bytes, "rb") as fh:
            data = fh.read()
    if data is None:
        return {"text": "", "fields": {}, "pages": 0, "method": "none"}

    ext = os.path.splitext(name)[1].lower()
    is_pdf = ext == ".pdf" or data[:5] == b"%PDF-"

    if is_pdf:
        text, pages = "", 0
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                pages = len(pdf.pages)
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:  # noqa: BLE001 — fall back to pypdf
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(data))
                pages = len(reader.pages)
                text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
            except Exception:  # noqa: BLE001
                text = ""
        if text.strip():
            fields = _extract_fields(text)
            tables = extract_tables(data)
            form_fields = extract_form_fields(text)
            if tables:
                fields["tables"] = tables
            if form_fields:
                fields["form_fields"] = form_fields
            return {"text": text.strip(), "fields": fields, "pages": pages, "method": "pdf-text"}
        # no text layer: real OCR of the scanned pages via poppler + tesseract
        ocr_text, ocr_pages = _ocr_scanned_pdf(data)
        if ocr_text:
            return {"text": ocr_text, "fields": _extract_fields(ocr_text),
                    "pages": ocr_pages or pages, "method": "pdf-ocr-scanned"}
        return {"text": "", "fields": {}, "pages": pages, "method": "pdf-scanned",
                "note": "Scanned PDF and OCR unavailable (install poppler + pytesseract)."}

    # image -> real OCR via tesseract
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(io.BytesIO(data)))
        if text.strip():
            return {"text": text.strip(), "fields": _extract_fields(text),
                    "pages": 1, "method": "tesseract-ocr"}
    except Exception as e:  # noqa: BLE001
        return {"text": "", "fields": {}, "pages": 0, "method": "ocr-error", "note": str(e)[:160]}
    return {"text": "", "fields": {}, "pages": 0, "method": "unreadable"}


# --------------------------------------------------------------------------- #
# 4) Document layout: tables, form key-value fields, scanned-page OCR
# --------------------------------------------------------------------------- #
_FORM_KV = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9 /._'\-]{1,40}?)\s*[:#]\s*(.+?)\s*$")
# KYC/AML form fields worth surfacing as structured key-values
_FORM_LABELS = ("account number", "account no", "iban", "sort code", "routing", "swift",
                "date of birth", "dob", "tax id", "ein", "ssn", "national id", "passport",
                "name", "legal name", "address", "registration", "incorporation", "lei",
                "beneficial owner", "ownership", "shareholder", "director", "officer",
                "closing balance", "opening balance", "statement date", "period")


def extract_tables(data: bytes) -> list[list[list[str]]]:
    """Real table extraction from a PDF (ruled tables via pdfplumber; text-strategy fallback).
    Returns a list of tables, each a list of cleaned rows."""
    import io
    tables: list[list[list[str]]] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                found = page.extract_tables() or []
                if not found:
                    found = page.extract_tables(
                        {"vertical_strategy": "text", "horizontal_strategy": "text"}) or []
                for tbl in found:
                    rows = []
                    for row in tbl:
                        cells = [(c or "").strip() for c in row]
                        if any(cells):
                            rows.append(cells)
                    if len(rows) >= 2:                         # header + >=1 data row
                        tables.append(rows)
    except Exception:  # noqa: BLE001
        pass
    return tables


def extract_form_fields(text: str) -> dict:
    """Extract `Label: value` form key-values an analyst expects from a KYC/onboarding doc."""
    out: dict = {}
    for line in (text or "").splitlines():
        m = _FORM_KV.match(line)
        if not m:
            continue
        label = m.group(1).strip().lower()
        value = m.group(2).strip()
        if 1 <= len(value) <= 120 and any(lbl in label for lbl in _FORM_LABELS):
            out[label] = value
    return out


def _ocr_scanned_pdf(data: bytes) -> tuple[str, int]:
    """Render a scanned (no-text-layer) PDF to images via poppler and OCR each page."""
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception:  # noqa: BLE001
        return "", 0
    try:
        images = convert_from_bytes(data, dpi=200)
    except Exception:  # noqa: BLE001 — poppler not installed / unreadable
        return "", 0
    parts = [pytesseract.image_to_string(im) for im in images]
    return "\n".join(parts).strip(), len(images)


def parse_edgar_facts(facts: dict) -> dict:
    """Parse SEC EDGAR companyfacts (XBRL us-gaap) into headline firmographics. Real, no key —
    the live fetch lives in tools_native; this is the (offline-testable) parser."""
    out: dict = {}
    gaap = ((facts or {}).get("facts") or {}).get("us-gaap") or {}
    out["company"] = (facts or {}).get("entityName")
    out["cik"] = (facts or {}).get("cik")

    from datetime import date
    _ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "40-F")

    def _full_year(v) -> bool:
        """A flow (income-statement) concept value covers a FULL fiscal year, not a quarter. A
        10-K reports both; they're told apart by the period duration. Entries without a start date
        (e.g. simplified fixtures) are accepted."""
        s, e = v.get("start"), v.get("end")
        if not s or not e:
            return True
        try:
            return (date.fromisoformat(e) - date.fromisoformat(s)).days >= 300
        except Exception:  # noqa: BLE001
            return True

    def annual(concept_names, *, flow, target_end=None):
        """Best annual value for a concept. For flow concepts only accept full-year periods (so a
        quarterly figure is never mistaken for the year); for instant (balance-sheet) concepts take
        the latest. When target_end is given, align to that fiscal period so revenue, net income and
        balance items all come from the SAME year."""
        # Pool across all equivalent tags (filers migrate between them — e.g. "Revenues" ->
        # "RevenueFromContractWithCustomerExcludingAssessedTax" after ASC 606), then pick the most
        # recent full-year value. Stopping at the first tag with any data would return a stale
        # period from a deprecated tag.
        cands = []
        for concept in concept_names:
            node = gaap.get(concept)
            if not node:
                continue
            for unit_vals in (node.get("units") or {}).values():
                for v in unit_vals:
                    if v.get("val") is None or v.get("form") not in _ANNUAL_FORMS:
                        continue
                    if flow and not _full_year(v):
                        continue
                    cands.append(v)
        if not cands:
            return None, None
        if target_end:
            exact = [c for c in cands if c.get("end") == target_end]
            if exact:
                cands = exact
        best = max(cands, key=lambda v: v.get("end", ""))
        return best.get("val"), best.get("end")

    rev, rev_end = annual(["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                           "SalesRevenueNet"], flow=True)
    ni, _ = annual(["NetIncomeLoss"], flow=True, target_end=rev_end)
    assets, _ = annual(["Assets"], flow=False, target_end=rev_end)
    equity, _ = annual(["StockholdersEquity"], flow=False, target_end=rev_end)
    emp, _ = annual(["EntityNumberOfEmployees", "NumberOfEmployees"], flow=False)
    if rev is not None:
        out["revenue"] = rev; out["revenue_period"] = rev_end
    if ni is not None:
        out["net_income"] = ni
    if assets is not None:
        out["total_assets"] = assets
    if equity is not None:
        out["stockholders_equity"] = equity
    if emp is not None:
        out["employees"] = emp
    return out


# --------------------------------------------------------------------------- #
# 5) Contact resolution heuristics (real, clearly-labelled — used when no API key)
# --------------------------------------------------------------------------- #
def email_candidates(name: str, domain: str) -> list[dict]:
    """Generate the common corporate email permutations for a name@domain with a confidence
    ranking. This is a real heuristic (what Hunter/Clearbit also start from), not sample data."""
    parts = [p for p in re.split(r"\s+", (name or "").strip().lower()) if p]
    if not parts or not domain:
        return []
    f, l = parts[0], parts[-1]
    fi, li = f[0], l[0]
    patterns = [
        (f"{f}.{l}", 0.82), (f"{fi}{l}", 0.74), (f"{f}{l}", 0.6), (f"{f}_{l}", 0.5),
        (f"{f}", 0.45), (f"{f}.{li}", 0.4), (f"{fi}.{l}", 0.4),
    ]
    seen, out = set(), []
    for local, conf in patterns:
        if local in seen:
            continue
        seen.add(local)
        out.append({"email": f"{local}@{domain}", "pattern": local, "confidence": conf})
    return out
