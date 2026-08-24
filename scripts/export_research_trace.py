"""Export GPU-Lab state and operational traces to a redacted Markdown report.

This is an observational export: it records persisted inputs, tool outcomes,
scientific decisions, events, runs, and artifacts. It deliberately does not
attempt to export private model chain-of-thought; use the structured decision
rationale and operator outputs persisted by Research OS instead.

Usage:
    uv run python scripts/export_research_trace.py --output trace.md
    uv run python scripts/export_research_trace.py --project-id UUID --output p.md
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from gpu_lab.config import Settings

_SECRET_KEY = re.compile(
    r"(token|secret|password|api[_-]?key|authorization|private[_-]?key|credential)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def redact(value: Any, key: str | None = None) -> Any:
    """Redact credentials recursively while preserving useful trace structure."""
    if key and _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _BEARER.sub(r"\1[REDACTED]", value)
    return value


def _json(value: Any) -> str:
    return json.dumps(redact(value), indent=2, sort_keys=True, default=str)


def _rows(conn, query: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(query, args)
        return list(cur.fetchall())


def _in_scope(row: dict[str, Any], start: datetime | None, end: datetime | None) -> bool:
    if not start and not end:
        return True
    value = row.get("created_at") or row.get("committed_at") or row.get("updated_at")
    if value is None:
        return True
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value = value.astimezone(UTC)
    return (start is None or value >= start) and (end is None or value < end)


def _filter_rows(
    rows: list[dict[str, Any]],
    start: datetime | None,
    end: datetime | None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    needle = query.casefold() if query else None
    return [
        row
        for row in rows
        if _in_scope(row, start, end)
        and (needle is None or needle in json.dumps(row, default=str).casefold())
    ]


def load_research(
    url: str,
    project_id: str | None,
    start: datetime | None = None,
    end: datetime | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    try:
        conn = psycopg.connect(url, row_factory=dict_row)
    except psycopg.OperationalError as exc:
        # Compose service names resolve only inside Docker. The compose file
        # publishes PostgreSQL on loopback for host-side exports.
        if "@postgres:" not in url:
            raise RuntimeError(f"Unable to connect to research PostgreSQL: {exc}") from exc
        host_url = url.replace("@postgres:", "@127.0.0.1:", 1)
        fallback_urls = [host_url]
        if "://" in host_url and "@" in host_url:
            fallback_urls.append(host_url.replace("research:@127.0.0.1:", "research:research@127.0.0.1:"))
        last_exc: Exception | None = None
        for candidate in fallback_urls:
            try:
                conn = psycopg.connect(candidate, row_factory=dict_row)
                break
            except psycopg.OperationalError as candidate_exc:
                last_exc = candidate_exc
        else:
            raise RuntimeError(
                "Research PostgreSQL is unreachable. Start Compose and ensure the local "
                "5432 port is published (docker compose up -d postgres)."
            ) from last_exc
    with conn:
        projects = _rows(
            conn,
            "SELECT id,name,question,state,created_at FROM research_projects "
            + ("WHERE id=%s " if project_id else "")
            + "ORDER BY created_at",
            (project_id,) if project_id else (),
        )
        if query:
            pattern = f"%{query}%"
            matching = _rows(
                conn,
                "SELECT DISTINCT project_id FROM research_objects "
                "WHERE data::text ILIKE %s "
                "UNION SELECT DISTINCT project_id FROM research_events "
                "WHERE payload::text ILIKE %s",
                (pattern, pattern),
            )
            matching_ids = {str(row["project_id"]) for row in matching}
            matching_ids.update(
                str(project["id"])
                for project in _filter_rows(projects, None, None, query)
            )
            projects = [project for project in projects if str(project["id"]) in matching_ids]
        ids = [str(p["id"]) for p in projects]
        if not ids:
            return {"projects": [], "objects": [], "edges": [], "events": [], "attempts": [], "versions": []}
        objects = _rows(conn, "SELECT id,project_id,kind,status,data,created_at FROM research_objects WHERE project_id=ANY(%s::uuid[]) ORDER BY created_at", (ids,))
        edges = _rows(conn, "SELECT source_id,target_id,relation,created_at FROM research_edges WHERE source_id=ANY(%s::uuid[]) OR target_id=ANY(%s::uuid[]) ORDER BY created_at", (ids, ids))
        events = _rows(conn, "SELECT id,project_id,event_type,subject_id,payload,created_at,committed_at,legacy_backfill FROM research_events WHERE project_id=ANY(%s::uuid[]) ORDER BY created_at", (ids,))
        attempts = _rows(conn, "SELECT experiment_id,idempotency_key,run_id,job_id,request_fingerprint,status,created_at,updated_at FROM research_execution_attempts WHERE experiment_id=ANY(%s::uuid[]) ORDER BY created_at", (ids,))
        versions = _rows(conn, "SELECT revision_id,object_id,project_id,kind,status,data,valid_from,committed_at,legacy_backfill FROM research_object_versions WHERE project_id=ANY(%s::uuid[]) ORDER BY revision_id", (ids,))
        return {
            "projects": projects,
            "objects": _filter_rows(objects, start, end),
            "edges": _filter_rows(edges, start, end),
            "events": _filter_rows(events, start, end),
            "attempts": _filter_rows(attempts, start, end),
            "versions": _filter_rows(versions, start, end),
        }


def load_operational(
    settings: Settings,
    start: datetime | None = None,
    end: datetime | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    path = settings.db_path
    if not path.exists():
        return {"instances": [], "jobs": [], "events": [], "audit": [], "database": str(path), "available": False}
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        def all_rows(table: str) -> list[dict[str, Any]]:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
        return {
            "instances": _filter_rows(all_rows("instances"), start, end, query),
            "jobs": _filter_rows(all_rows("jobs"), start, end, query),
            "events": _filter_rows(all_rows("events"), start, end, query),
            "audit": _filter_rows(all_rows("audit_log"), start, end, query),
            "database": str(path),
            "available": True,
        }
    finally:
        conn.close()


def render(data: dict[str, Any], operational: dict[str, Any], output: Path) -> None:
    lines = [
        "# GPU-Lab Research Trace Export",
        "",
        f"- Exported at: `{datetime.now(UTC).isoformat()}`",
        "- Scope: persisted PostgreSQL scientific state and local operational records.",
        "- Privacy: credentials and bearer values are redacted.",
        "- Reasoning boundary: this contains structured rationale, prompts/arguments where persisted, tool outputs, and events—not private hidden chain-of-thought.",
        "",
    ]
    for title, key in (("Projects", "projects"), ("Scientific objects", "objects"), ("Scientific edges", "edges"), ("Research events", "events"), ("Execution attempts", "attempts"), ("Object version history", "versions")):
        rows = data[key]
        lines += [f"## {title}", "", f"Count: **{len(rows)}**", ""]
        for row in rows:
            ident = row.get("id") or row.get("revision_id") or row.get("experiment_id") or row.get("source_id")
            label = row.get("kind") or row.get("event_type") or row.get("status") or "record"
            lines += [f"### `{label}` — `{ident}`", "", "```json", _json(row), "```", ""]
    lines += ["## Operational records", "", f"Database: `{operational['database']}` (available: `{operational['available']}`)", ""]
    for title, key in (("MCP audit log", "audit"), ("Jobs", "jobs"), ("Worker events", "events"), ("Instances", "instances")):
        rows = operational[key]
        lines += [f"### {title}", "", f"Count: **{len(rows)}**", "", "```json", _json(rows), "```", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("trace_export.md"))
    parser.add_argument("--project-id")
    parser.add_argument("--from", dest="start", help="Inclusive UTC date/time (ISO-8601)")
    parser.add_argument("--to", dest="end", help="Exclusive UTC date/time (ISO-8601)")
    parser.add_argument("--query", help="Case-insensitive text filter, such as 'point cloud'")
    args = parser.parse_args()
    try:
        start = datetime.fromisoformat(args.start) if args.start else None
        end = datetime.fromisoformat(args.end) if args.end else None
    except ValueError as exc:
        raise SystemExit(f"Invalid --from/--to timestamp: {exc}") from exc
    if start and start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end and end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    settings = Settings()
    if not settings.gpu_lab_research_database_url:
        raise SystemExit("GPU_LAB_RESEARCH_DATABASE_URL is not configured")
    data = load_research(
        settings.gpu_lab_research_database_url,
        args.project_id,
        start.astimezone(UTC) if start else None,
        end.astimezone(UTC) if end else None,
        args.query,
    )
    render(data, load_operational(settings, start, end, args.query), args.output)
    print(f"Wrote {args.output} ({len(data['objects'])} scientific objects, {len(data['events'])} research events)")


if __name__ == "__main__":
    main()

