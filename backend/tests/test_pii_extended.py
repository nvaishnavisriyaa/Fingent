"""
PII redaction now covers the identifiers the platform itself parses out of KYC/financial documents
(IBAN, account number, DOB, passport/national/tax id) — previously extracted but left in the clear.
Legitimate financial figures and ordinary words must NOT be redacted.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fingent.middleware import redact_pii, redact_obj


def test_redacts_iban_account_dob_passport():
    red, found = redact_pii(
        "IBAN GB29NWBK60161331926819; Account No ACC-00123; DOB: 1980-04-12; Passport No A1234567")
    assert "GB29NWBK60161331926819" not in red
    assert "ACC-00123" not in red
    assert "1980-04-12" not in red
    assert "A1234567" not in red
    assert {"iban", "account", "dob", "passport"} <= set(found)


def test_does_not_redact_financial_figures_or_plain_words():
    red, found = redact_pii("revenue 62000000 ebitda 9300000; status active; current assets 18000000")
    assert red == "revenue 62000000 ebitda 9300000; status active; current assets 18000000"
    assert found == []


def test_redact_obj_scrubs_nested_tool_output():
    obj = {"account_number": "Account No 998877665544", "ratios": {"current_ratio": 1.6}}
    clean = redact_obj(obj)
    assert "998877665544" not in str(clean)
    assert clean["ratios"]["current_ratio"] == 1.6     # real numbers preserved
