"""
Depth for the GTM/enrichment + KYC-document tools (offline, deterministic):
  * PDF table extraction + form key-value fields + scanned-page OCR (poppler+tesseract)
  * SEC EDGAR company-facts -> real public-company firmographics
  * PEP screening via the real entity-resolution engine
  * honest email-pattern heuristic for contacts; sample MCP tools labelled truthfully
"""
import io, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import realtools as rt
import fingent.tools_native as t


def _ruled_pdf() -> bytes:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=letter)
    rows = [("Beneficial Owner", "Role", "Ownership %"), ("Jane Doe", "Director", "55"),
            ("John Roe", "Officer", "30"), ("Acme Holdings Ltd", "Entity", "15")]
    x0, y, colw = 72, 760, [170, 110, 120]
    for row in rows:
        x = x0
        for i, cell in enumerate(row):
            c.drawString(x + 3, y + 5, cell); x += colw[i]
        y -= 24
    top, bottom = 782, 760 - 24 * len(rows) + 19
    x = x0
    for w in colw + [0]:
        c.line(x, top, x, bottom); x += w
    yy = 782
    for _ in range(len(rows) + 1):
        c.line(x0, yy, x0 + sum(colw), yy); yy -= 24
    c.drawString(72, 300, "Account Number: 12345678")
    c.drawString(72, 282, "Tax ID: 12-3456789")
    c.drawString(72, 264, "Date of Birth: 1980-04-12")
    c.save(); return buf.getvalue()


def test_pdf_table_and_form_extraction():
    out = rt.extract_document(_ruled_pdf(), "kyc.pdf")
    tables = out["fields"]["tables"]
    assert tables and tables[0][0] == ["Beneficial Owner", "Role", "Ownership %"]
    assert ["Jane Doe", "Director", "55"] in tables[0]              # beneficial-owner row
    ff = out["fields"]["form_fields"]
    assert ff["account number"] == "12345678" and ff["tax id"] == "12-3456789"


def test_scanned_pdf_is_ocred_via_poppler():
    import shutil
    if not (shutil.which("tesseract") and shutil.which("pdftoppm")):
        import pytest
        pytest.skip("tesseract/poppler not installed (CI installs tesseract-ocr + poppler-utils)")
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1000, 220), "white")
    ImageDraw.Draw(img).text((20, 90), "SCANNED ONBOARDING FORM  Tax ID 98-7654321", fill="black")
    ib = io.BytesIO(); img.save(ib, "PDF")          # image-only PDF (no text layer)
    out = rt.extract_document(ib.getvalue(), "scan.pdf")
    assert out["method"] == "pdf-ocr-scanned"
    assert "SCANNED" in out["text"].upper()


def test_edgar_companyfacts_parser():
    facts = {"entityName": "ACME INC", "cik": 99, "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"val": 62_000_000, "end": "2025-12-31", "form": "10-K"},
            {"val": 50_000_000, "end": "2024-12-31", "form": "10-K"}]}},
        "NetIncomeLoss": {"units": {"USD": [{"val": 9_000_000, "end": "2025-12-31", "form": "10-K"}]}},
        "Assets": {"units": {"USD": [{"val": 120_000_000, "end": "2025-12-31", "form": "10-K"}]}}}}}
    p = rt.parse_edgar_facts(facts)
    assert p["revenue"] == 62_000_000 and p["revenue_period"] == "2025-12-31"   # latest 10-K
    assert p["net_income"] == 9_000_000 and p["total_assets"] == 120_000_000


def test_email_heuristic_is_ranked():
    cands = rt.email_candidates("Jane Doe", "acme.com")
    assert cands[0] == {"email": "jane.doe@acme.com", "pattern": "jane.doe", "confidence": 0.82}
    assert all(0 < c["confidence"] <= 1 for c in cands)


def test_pep_check_offline_uses_entity_resolution(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    hit = t.pep_check("Vladimir Putin")
    assert hit["pep"] is True and hit["match_type"] in ("exact", "strong")
    clean = t.pep_check("Ordinary Person")
    assert clean["pep"] is False


def test_resolve_contact_offline_is_real_heuristic(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    r = t.resolve_contact("Jane Doe", "acme.com")
    assert r["source"] == "computed:email-heuristic"
    assert r["email"] == "jane.doe@acme.com" and len(r["candidates"]) >= 3


def test_find_persona_offline_does_not_fabricate_people(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    r = t.find_persona("Acme")
    assert r["source"] == "computed:persona-heuristic"
    assert r["personas"] == [] and "Chief Financial Officer" in r["target_titles"]


def test_sample_mcp_tools_are_labelled_not_fake_live():
    q = t.mcp_bloomberg_quote("AAPL")
    assert q["source"] == "demo:sample-mcp" and q["price"] is None
    e = t.mcp_send_email(to="x@y.com")
    assert e["source"] == "demo:sample-mcp" and e["sent"] is False


def test_ocr_extract_tool_surfaces_tables_and_forms():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(_ruled_pdf()); path = fh.name
    try:
        r = t.ocr_extract(path)
        assert r["source"] == "live:local-extract"
        assert "tables" in r["fields"] and "form_fields" in r["fields"]
    finally:
        os.unlink(path)
