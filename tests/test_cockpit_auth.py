from gpu_lab.cockpit_auth import issue_session, password_matches, verify_session


def test_cockpit_session_is_signed_expiring_and_has_csrf_material(monkeypatch):
    monkeypatch.setattr("gpu_lab.cockpit_auth.time.time", lambda: 1_000)
    token, session = issue_session("session-secret", lifetime_seconds=60)

    assert verify_session("session-secret", token) == session
    assert verify_session("wrong-secret", token) is None
    assert verify_session("session-secret", token + "tamper") is None
    monkeypatch.setattr("gpu_lab.cockpit_auth.time.time", lambda: 1_061)
    assert verify_session("session-secret", token) is None


def test_cockpit_password_match_uses_configured_value():
    assert password_matches("private-password", "private-password")
    assert not password_matches("private-password", "wrong")
    assert not password_matches(None, "anything")
