import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "export_research_trace", Path(__file__).parents[1] / "scripts" / "export_research_trace.py"
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
redact = _MODULE.redact


def test_redact_removes_credentials_and_bearer_values():
    value = redact(
        {
            "api_key": "secret",
            "nested": {"authorization": "Bearer abc.def"},
            "message": "Authorization: Bearer xyz",
            "safe": "keep",
        }
    )
    assert value["api_key"] == "[REDACTED]"
    assert value["nested"]["authorization"] == "[REDACTED]"
    assert "xyz" not in value["message"]
    assert value["safe"] == "keep"
