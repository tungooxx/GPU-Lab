"""Pure, read-only v3.6 research-portfolio scheduling projection.

The shadow scheduler intentionally allocates only existing authoritative READY
WorkItems. It never creates a hypothesis, WorkItem, lease, or scientific fact.
"""

from __future__ import annotations

from typing import Any


class ResearchPortfolioV36Shadow:
    """Deterministic shadow scheduler for review before staged activation."""

    VERSION = "v3.6-shadow"

    @classmethod
    def project(
        cls,
        workers: list[dict[str, Any]],
        ready_work: list[dict[str, Any]],
        branch_coverage: dict[str, dict[str, Any]],
        objective_global_blocks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assignments: list[dict[str, Any]] = []
        used_work_ids: set[str] = set()
        projected_branch_ids: set[str] = set()

        def rank(item: dict[str, Any]) -> tuple[int, int, float, str]:
            branch_id = str(item.get("branch_id") or "")
            branch = branch_coverage.get(branch_id, {})
            # Branch work takes precedence over unscoped operational work, but
            # only when that branch has no active worker and is not already
            # allocated by this shadow pass.
            concentration = int(branch.get("active_worker_count", 0)) if branch else 10_000
            projected = 1 if branch_id and branch_id in projected_branch_ids else 0
            return (concentration, projected, -float(item.get("priority") or 0), str(item["id"]))

        for worker in workers:
            if worker.get("availability_state") not in {"AVAILABLE", "IDLE"}:
                continue
            if objective_global_blocks:
                assignments.append({
                    "worker_id": worker["id"],
                    "suggested_work_item_id": None,
                    "reason": "OBJECTIVE_GLOBAL_DEPENDENCY_BLOCK",
                    "global_blocks": objective_global_blocks,
                })
                continue

            candidates = [
                item for item in ready_work
                if str(item["id"]) not in used_work_ids
                and not (
                    item.get("branch_id")
                    and int(branch_coverage.get(str(item["branch_id"]), {}).get("active_worker_count", 0)) > 0
                )
                and not (
                    item.get("branch_id")
                    and str(item["branch_id"]) in projected_branch_ids
                )
            ]
            if not candidates:
                assignments.append({
                    "worker_id": worker["id"],
                    "suggested_work_item_id": None,
                    "reason": "IDLE_NO_EXISTING_ACTIONABLE_WORK",
                })
                continue

            chosen = min(candidates, key=rank)
            work_id = str(chosen["id"])
            branch_id = chosen.get("branch_id")
            used_work_ids.add(work_id)
            if branch_id:
                projected_branch_ids.add(str(branch_id))
            assignments.append({
                "worker_id": worker["id"],
                "suggested_work_item_id": work_id,
                "branch_id": branch_id,
                "reason": "EXISTING_READY_CANONICAL_WORK",
                "dependency_scope": chosen.get("dependency_scope"),
                "branch_coverage": branch_coverage.get(str(branch_id or ""), {}),
            })

        return {
            "projection_version": cls.VERSION,
            "assignments": assignments,
            "unassigned_ready_work_item_ids": [
                str(item["id"]) for item in ready_work if str(item["id"]) not in used_work_ids
            ],
            "objective_global_blocks": objective_global_blocks,
            "planner_action": "OBJECTIVE_GLOBAL_BLOCK" if objective_global_blocks else (
                "DO_NOT_CREATE_WORK" if not ready_work else "CLAIM_EXISTING_ONLY"
            ),
            "mutated": False,
        }
