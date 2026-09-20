from guardrails_rag import secret_scrubber

ANNEX_D_SNIPPET = """
API Master Key (Sandbox) X-API-KEY: ak_live_7x89fjsdf834hfsd89234hfksdf89
AWS_ACCESS_KEY_ID: AKIAIOSFODNN7EXAMPLE
DB_PASS: P@ssw0rd_Chimera_2026!
SSH_PRIVATE_KEY: -----BEGIN RSA PRIVATE KEY-----
MIIEpQIBAAKCAQEA3Tz2Y9p...
-----END RSA PRIVATE KEY-----
Contact s.jenkins@zenith-data-sandbox.net or 192.168.105.42
SSN 999-12-3456
"""


def test_scrub_redacts_every_secret_class():
    result = secret_scrubber.scrub(ANNEX_D_SNIPPET)
    assert result.had_secrets
    assert "AWS_ACCESS_KEY" in result.hits
    assert "GENERIC_API_KEY" in result.hits
    assert "SSH_PRIVATE_KEY" in result.hits
    assert "PASSWORD_ASSIGNMENT" in result.hits
    assert "EMAIL" in result.hits
    assert "IPV4" in result.hits
    assert "SSN" in result.hits
    assert "AKIAIOSFODNN7EXAMPLE" not in result.text
    assert "ak_live_7x89fjsdf834hfsd89234hfksdf89" not in result.text
    assert "BEGIN RSA PRIVATE KEY" not in result.text


def test_scrub_leaves_clean_text_untouched():
    text = "Zenith Data Solutions must complete the Vendor Security Questionnaire."
    result = secret_scrubber.scrub(text)
    assert not result.had_secrets
    assert result.text == text
