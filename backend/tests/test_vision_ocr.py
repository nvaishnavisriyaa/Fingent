import os, sys, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fingent.tools_native as t

class _R:
    ok = True
    def __init__(self, payload): self._p = payload
    def json(self): return self._p

def test_vision_ocr_parses_json(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("OCR_API_URL", raising=False)
    captured = {}
    def fake_post(url, json=None, headers=None, timeout=None, **k):
        captured["url"] = url; captured["model"] = json["model"]
        content = '{"text": "ACME Bank Statement closing balance 1204332", "fields": {"closing_balance": 1204332}}'
        return _R({"choices": [{"message": {"content": content}}]})
    monkeypatch.setattr(t, "_post", fake_post)
    r = t.ocr_extract("https://example.com/statement.png")
    assert r["source"].startswith("live:Groq vision")
    assert r["fields"]["closing_balance"] == 1204332
    assert "closing balance" in r["text"].lower()
    assert "groq.com" in captured["url"]

def test_bare_filename_is_unavailable_not_fabricated(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("OCR_API_URL", raising=False)
    # a filename with no readable bytes can't be OCR'd -> honest, no 1,204,332 mock
    r = t.ocr_extract("financials.pdf")
    assert r["source"] == "unavailable"
    assert "1204332" not in str(r) and "1,204,332" not in str(r)

def test_offline_keeps_sample(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    assert t.ocr_extract("x.pdf")["source"] == "mock"
