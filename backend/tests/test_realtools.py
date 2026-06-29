"""
Depth, not stubs: the real analytical engines behind the AML/KYC tools.
  * sanctions entity resolution (fuzzy match + scoring + classification)
  * adverse-media NLP (category classification, negation, aggregate risk)
  * real document extraction (PDF text + image OCR), exercised end-to-end
All offline and deterministic.
"""
import io, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import realtools as rt
import fingent.tools_native as t


# ---- 1) sanctions entity resolution ---------------------------------------- #
SDN = ["PETROV, Oleg Vladimirovich", "IVANOV, Sergei", "AL-EXAMPLE Holding LLC",
       "SMITH, John A", "DOE, Jane"]


def test_entity_resolution_matches_reordered_and_partial_names():
    r = rt.screen_names("Oleg Petrov", SDN)
    assert r["ofac_hit"] is True
    assert r["candidates"][0]["name"] == "PETROV, Oleg Vladimirovich"
    assert r["candidates"][0]["match"] in ("exact", "strong")
    assert r["best_score"] >= 0.85


def test_entity_resolution_no_false_positive_on_clean_name():
    r = rt.screen_names("Maria Gonzalez", SDN)
    assert r["ofac_hit"] is False and r["best_score"] < 0.85


def test_match_scoring_is_ordered_and_bounded():
    assert rt.match_score("John Smith", "SMITH, John A") > rt.match_score("John Smith", "DOE, Jane")
    assert 0.0 <= rt.match_score("a", "b") <= 1.0


# ---- 2) adverse-media NLP -------------------------------------------------- #
def test_adverse_media_categorizes_and_scores():
    a = rt.score_adverse("Acme", [
        "Acme CEO charged in money laundering and wire fraud scheme",
        "Regulator opens investigation into Acme over sanctions evasion",
        "Acme launches a new savings account"])
    assert a["risk_band"] == "high" and a["risk_score"] >= 60
    assert "financial_crime" in a["categories"] and "sanctions" in a["categories"]
    assert a["adverse_hits"] == 2          # the benign headline is not flagged


def test_adverse_media_handles_negation():
    pos = rt.classify_headline("Acme charged with bribery")
    neg = rt.classify_headline("Acme cleared of bribery charges")
    assert neg["severity"] < pos["severity"]    # exoneration is down-weighted


# ---- 3) real document extraction (OCR) ------------------------------------- #
def _make_pdf() -> bytes:
    from reportlab.pdfgen import canvas
    buf = io.BytesIO(); c = canvas.Canvas(buf)
    for ln, y in [("ACME CORP - Bank Statement", 760),
                  ("Account Number: GB29NWBK60161331926819", 740),
                  ("Statement date: 2026-03-31", 720),
                  ("Revenue: $62,000,000  EBITDA: $9,300,000", 700),
                  ("Closing balance: $1,204,332.55", 680)]:
        c.drawString(72, y, ln)
    c.save(); return buf.getvalue()


def test_real_pdf_text_extraction_and_field_parsing():
    out = rt.extract_document(_make_pdf(), "statement.pdf")
    assert out["method"] == "pdf-text" and out["pages"] == 1
    assert "ACME CORP" in out["text"]
    f = out["fields"]
    assert f["closing_balance"] == 1204332.55
    assert f["account_number"] == "GB29NWBK60161331926819"
    assert "2026-03-31" in f["dates_found"]


def test_real_image_ocr_with_tesseract():
    import shutil
    if not shutil.which("tesseract"):
        import pytest
        pytest.skip("tesseract binary not installed (CI installs tesseract-ocr)")
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 80), "white")
    ImageDraw.Draw(img).text((10, 30), "INVOICE TOTAL 4827 USD", fill="black")
    b = io.BytesIO(); img.save(b, "PNG")
    out = rt.extract_document(b.getvalue(), "invoice.png")
    assert out["method"] == "tesseract-ocr"
    assert "INVOICE" in out["text"].upper()


def test_ocr_extract_tool_uses_real_local_extraction_on_a_file():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(_make_pdf()); path = fh.name
    try:
        r = t.ocr_extract(path)
        assert r["source"] == "live:local-extract" and r["method"] == "pdf-text"
        assert r["fields"]["closing_balance"] == 1204332.55
    finally:
        os.unlink(path)


# ---- 4) the AML tools surface the real engines ----------------------------- #
def test_ofac_screen_returns_scored_resolution(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")     # offline -> resolver runs on labelled fixture
    r = t.ofac_screen("Oleg Petrov")
    assert r["ofac_hit"] is True and r["match_type"] in ("exact", "strong")
    assert r["matches"] and "score" in r["matches"][0]


def test_adverse_media_tool_offline_shape(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    r = t.adverse_media_search("Acme")
    assert "risk_score" in r and "categories" in r
