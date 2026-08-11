import hmac
import os
from pathlib import Path

import uvicorn
from pydantic import BaseModel, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .errors import GPUError
from .literature import PaperQALiteratureProvider


class WorkerRequest(BaseModel):
    source: str | None = Field(default=None, max_length=4096)
    query: str | None = Field(default=None, max_length=20000)
    question: str | None = Field(default=None, max_length=20000)
    papers: list[str] | None = Field(default=None, max_length=100)
    filters: dict | None = None
    paper_id: str | None = Field(default=None, max_length=1000)
    request: str | None = Field(default=None, max_length=20000)


WORKER_TOKEN = os.environ.get("GPU_LAB_LITERATURE_WORKER_TOKEN", "")


def _configured_retries() -> int:
    try:
        return int(os.environ.get("GPU_LAB_PAPERQA_MAX_RETRIES", "2"))
    except ValueError:
        return -1


provider = PaperQALiteratureProvider(
    Path(os.environ.get("GPU_LAB_PAPERQA_DIRECTORY", "/papers")),
    model=os.environ.get("GPU_LAB_PAPERQA_MODEL"),
    base_url=os.environ.get("GPU_LAB_PAPERQA_BASE_URL"),
    embedding_model=os.environ.get("GPU_LAB_PAPERQA_EMBEDDING_MODEL"),
    max_retries=_configured_retries(),
)


def _required(value, field: str):
    if value in (None, ""):
        raise GPUError("INVALID_LITERATURE_REQUEST", f"{field} is required")
    return value


async def dispatch(request: Request) -> JSONResponse:
    operation = request.path_params["operation"]
    if operation == "health":
        api_key_configured = bool(os.environ.get("OPENAI_API_KEY"))
        configuration_error = provider.configuration_error()
        return JSONResponse(
            {
                "result": {
                    "provider": "paperqa",
                    "status": (
                        "invalid_configuration"
                        if configuration_error
                        else "ready"
                        if api_key_configured
                        else "needs_credentials"
                    ),
                    "api_key_configured": api_key_configured,
                    "custom_endpoint": bool(provider.model or provider.base_url),
                    "model": provider.model,
                    "embedding_model": provider.embedding_model,
                    "max_retries": provider.max_retries,
                    "configuration_error": configuration_error,
                }
            }
        )
    authorization = request.headers.get("authorization", "")
    supplied = authorization.removeprefix("Bearer ").encode("utf-8", "replace")
    if not WORKER_TOKEN or not hmac.compare_digest(supplied, WORKER_TOKEN.encode()):
        return JSONResponse({"error": {"type": "UNAUTHORIZED", "message": "Unauthorized"}}, 401)
    try:
        body = WorkerRequest.model_validate(await request.json())
        if operation == "ingest":
            result = await provider.ingest(_required(body.source, "source"))
        elif operation == "search":
            result = await provider.search(_required(body.query, "query"), body.filters)
        elif operation == "retrieve-evidence":
            result = await provider.retrieve_evidence(_required(body.query, "query"), body.papers)
        elif operation == "inspect-document":
            result = await provider.inspect_document(
                _required(body.paper_id, "paper_id"), _required(body.request, "request")
            )
        elif operation == "ask":
            result = await provider.ask(_required(body.question, "question"), body.papers)
        else:
            raise GPUError("UNKNOWN_LITERATURE_OPERATION", operation)
        if isinstance(result, BaseModel):
            result = result.model_dump(mode="json")
        return JSONResponse({"result": result})
    except (GPUError, ValidationError) as exc:
        if isinstance(exc, GPUError):
            error = exc.response()["error"]
        else:
            error = {"type": "INVALID_LITERATURE_REQUEST", "message": str(exc)}
        return JSONResponse({"error": error}, 400)
    except Exception:  # noqa: BLE001 - sanitize all third-party failures at the worker boundary
        return JSONResponse(
            {
                "error": {
                    "type": "LITERATURE_PROVIDER_FAILURE",
                    "message": "The literature provider failed without changing scientific state",
                    "retryable": True,
                }
            },
            502,
        )


app = Starlette(routes=[Route("/{operation}", dispatch, methods=["POST"])])


def main() -> None:
    if not WORKER_TOKEN:
        raise RuntimeError("GPU_LAB_LITERATURE_WORKER_TOKEN is required")
    # Container-internal bind; Compose does not publish port 8010 to the host.
    uvicorn.run(app, host="0.0.0.0", port=8010, access_log=False)


if __name__ == "__main__":
    main()
