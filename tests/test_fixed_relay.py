from gpu_lab.fixed_relay import MAX_REQUEST_BODY, RelayHandler


class UnreadableBody:
    def read(self, _length):
        raise AssertionError("oversized body must not be read")


class TimedOutBody:
    def read(self, _length):
        raise TimeoutError


def test_fixed_relay_rejects_oversized_content_length_before_reading():
    handler = RelayHandler.__new__(RelayHandler)
    handler.headers = {"Content-Length": str(MAX_REQUEST_BODY + 1)}
    handler.rfile = UnreadableBody()
    errors = []
    handler.send_error = lambda code, message: errors.append((code, message))

    handler._forward()

    assert errors == [(413, "request body exceeds 1 MiB relay limit")]


def test_fixed_relay_returns_request_timeout_for_partial_body():
    handler = RelayHandler.__new__(RelayHandler)
    handler.headers = {"Content-Length": "1"}
    handler.rfile = TimedOutBody()
    errors = []
    handler.send_error = lambda code, message: errors.append((code, message))

    handler._forward()

    assert errors == [(408, "request body timeout")]
