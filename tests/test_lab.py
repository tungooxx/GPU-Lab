import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpu_lab.cockpit import CockpitController
from gpu_lab.errors import GPUError
from gpu_lab.lab import ACTIVE_WORK_STATUSES, LabController
from gpu_lab.research import ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_two_workers_claim_distinct_work_and_messages_are_not_evidence():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project = store.project_create(f"lab-two-workers-{time.time_ns()}", "Shared coordination")
    project_id = project["project_id"]
    first = lab.join(None, "worker-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "worker-b", "CODEX", project_id)
    first_worker, second_worker = first["worker"]["id"], second["worker"]["id"]
    work_a = lab.create_work(project_id, "LITERATURE", "Read evidence", "Independent retrieval", "LITERATURE_RESEARCHER", first_worker, created_session_id=first["session_id"])
    work_b = lab.create_work(project_id, "REVIEW", "Review result", "Independent review", "ADVERSARIAL_REVIEWER", second_worker, created_session_id=second["session_id"])

    lab.claim_work(work_a["id"], first_worker, first["session_id"])
    assert lab.start_work(work_a["id"], first_worker, first["session_id"])["status"] == "RUNNING"
    state_b = lab.state_get(project_id, second["session_id"])
    assert {item["id"] for item in state_b["running_work_items"]} == {work_a["id"]}
    with pytest.raises(GPUError, match="LAB_WORK_NOT_CLAIMABLE"):
        lab.claim_work(work_a["id"], second_worker, second["session_id"])
    assert lab.claim_work(work_b["id"], second_worker, second["session_id"])["id"] == work_b["id"]


    lab.message_send(project_id, first_worker, first["session_id"], "SHARE_FINDING", "Opinion", "H1 is definitely correct")
    assert lab.message_list(project_id, second_worker)
    assert store.objects_list(project_id, "Hypothesis", limit=None) == []


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_dependency_requires_explicit_status_or_exists_only_marker():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project = store.project_create(f"lab-dependency-status-{time.time_ns()}", "Dependency validation")
    worker = lab.join(None, "dependency-status-worker", "CODEX", project["project_id"])
    with pytest.raises(GPUError) as exc_info:
        lab.create_work(
            project["project_id"], "VALIDATION", "Missing status", "Must not become ready.",
            "VALIDATOR", worker["worker"]["id"], created_session_id=worker["session_id"],
            dependencies=[{"target_type": "RESEARCH_OBJECT", "target_id": str(uuid.uuid4())}],
        )
    assert exc_info.value.error_type == "LAB_DEPENDENCY_REQUIRED_STATUS_REQUIRED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_dependency_target_type_is_never_inferred_from_a_uuid():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project = store.project_create(f"lab-dependency-type-{time.time_ns()}", "Typed dependency")
    joined = lab.join(None, "typed-dependency-worker", "CODEX", project["project_id"])
    with pytest.raises(GPUError, match="LAB_DEPENDENCY_TARGET_TYPE_REQUIRED"):
        lab.create_work(project["project_id"], "REVIEW", "Type required", "No inference", "REVIEWER", joined["worker"]["id"], created_session_id=joined["session_id"], dependencies=[{"target_id": str(uuid.uuid4()), "required_statuses": ["COMPLETED"]}])


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v355_objective_is_versioned_and_proposals_merge_into_existing_work():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v355-proposal-{time.time_ns()}", "v3.5.5 proposal arbitration")["project_id"]
    joined = lab.join(None, "v355-worker", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = lab.canonical_objective_create(
        project_id, "SCIENTIFIC", "Inspect E201", "Does E201 satisfy the frozen metric contract?",
    )
    assert objective["version"] == 1
    assert lab.canonical_objective_create(
        project_id, "SCIENTIFIC", "Inspect E201", "Does E201 satisfy the frozen metric contract?",
    )["id"] == objective["id"]
    with pytest.raises(GPUError, match="CANONICAL_OBJECTIVE_VERSION_TRANSITION_REQUIRED"):
        lab.canonical_objective_create(project_id, "SCIENTIFIC", "Inspect E201", "A different question")

    authority = lab.authority_key(project_id, "E201", "v4", "RESULT_INSPECTION")
    canonical = lab.create_work(
        project_id, "REVIEW", "Inspect E201", "Canonical review", "RESULT_INSPECTOR", worker_id,
        created_session_id=session_id, authority_key=authority, authority_status="AUTHORITATIVE",
        canonical_subject_version="v4", subject_id="E201", canonical_objective_id=objective["id"],
    )
    proposal = lab.work_propose(
        project_id, worker_id, session_id, "RESULT_INSPECTION", "RESULT_INSPECTOR", "same inspection",
        canonical_objective_id=objective["id"], target_id="E201", authority_key_hint=authority,
    )
    assert proposal["status"] == "MERGED_INTO_EXISTING"
    assert proposal["canonical_work_item_id"] == canonical["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_branch_identity_is_structured_for_work_and_proposals():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v36-branch-link-{time.time_ns()}", "v3.6 branch linkage")["project_id"]
    joined = lab.join(None, "v36-branch-worker", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = lab.canonical_objective_create(
        project_id, "SCIENTIFIC", "Residual mechanism", "Which mechanism explains residual instability?",
    )
    branch = lab.hypothesis_branch_create(
        project_id, objective["id"], question_id="Q-residual", hypothesis_ids=["H-anchor"],
        mechanistic_niche_id="anchor-state", scientific_distance="NEAR",
    )
    work = lab.create_work(
        project_id, "ANALYSIS", "Inspect anchor state", "Bounded branch analysis", "RESULT_INSPECTOR",
        worker_id, created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=branch["id"],
    )
    proposal = lab.work_propose(
        project_id, worker_id, session_id, "ANALYSIS", "RESULT_INSPECTOR", "Independent branch follow-up",
        hypothesis_branch_id=branch["id"],
    )
    assert work["branch_id"] == branch["id"]
    assert proposal["hypothesis_branch_id"] == branch["id"]
    assert proposal["canonical_objective_id"] == objective["id"]
    coverage = lab.branch_coverage_get(project_id)
    assert coverage[0]["branch_id"] == branch["id"]
    assert coverage[0]["active_work_item_ids"] == [work["id"]]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_branch_cannot_cross_objective_boundary():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v36-branch-scope-{time.time_ns()}", "v3.6 branch scope")["project_id"]
    joined = lab.join(None, "v36-scope-worker", "CODEX", project_id)
    first = lab.canonical_objective_create(project_id, "SCIENTIFIC", "First", "First question")
    second = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Second", "Second question")
    branch = lab.hypothesis_branch_create(project_id, first["id"])
    with pytest.raises(GPUError, match="HYPOTHESIS_BRANCH_OBJECTIVE_MISMATCH"):
        lab.create_work(
            project_id, "ANALYSIS", "Wrong branch", "Must not cross objectives", "RESULT_INSPECTOR",
            joined["worker"]["id"], created_session_id=joined["session_id"],
            canonical_objective_id=second["id"], branch_id=branch["id"],
        )


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_refuting_branch_releases_live_lease_and_worker_ownership():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v36-branch-retire-{time.time_ns()}", "v3.6 branch retirement")["project_id"]
    joined = lab.join(None, "v36-branch-retire-worker", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Refutation", "Which branch survives?")
    branch = lab.hypothesis_branch_create(project_id, objective["id"])
    portfolio = lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    assert portfolio["active_branch_ids"] == [branch["id"]]
    work = lab.create_work(
        project_id, "ANALYSIS", "Live branch analysis", "Must be released when branch is refuted.",
        "RESULT_INSPECTOR", worker_id, created_session_id=session_id,
        canonical_objective_id=objective["id"], branch_id=branch["id"],
    )
    claimed = lab.claim_work(work["id"], worker_id, session_id)
    assert lab.start_work(work["id"], worker_id, session_id)["status"] == "RUNNING"

    transitioned = lab.hypothesis_branch_transition(
        branch["id"], worker_id, session_id, "REFUTED", "The discriminating prediction failed.",
    )

    assert transitioned["retired_descendants"] == 1
    retired = lab.work_get(work["id"])
    assert retired["status"] == "INVALIDATED"
    assert retired["lease_id"] is None
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT released_at,release_reason FROM lab_work_leases WHERE id=%s", (claimed["lease_id"],))
        lease = cur.fetchone()
        cur.execute("SELECT current_work_item_id,status FROM research_worker_sessions WHERE id=%s", (session_id,))
        session = cur.fetchone()
        cur.execute("SELECT availability_state FROM research_workers WHERE id=%s", (worker_id,))
        worker = cur.fetchone()
    assert lease["released_at"] is not None
    assert lease["release_reason"] == "BRANCH_RETIRED"
    assert session["current_work_item_id"] is None
    assert session["status"] == "ACTIVE"
    assert worker["availability_state"] == "AVAILABLE"
    refreshed_portfolio = lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    assert refreshed_portfolio["active_branch_ids"] == []
    assert refreshed_portfolio["status"] == "RESOLVED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v355_gate_supersession_releases_live_lease_and_worker_ownership():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v355-gate-retire-{time.time_ns()}", "gate retirement")["project_id"]
    joined = lab.join(None, "v355-gate-retire-worker", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    old_gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "E-gate-retire", "v1", worker_id, session_id)
    successor_gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "E-gate-retire", "v2", worker_id, session_id)
    work = lab.gate_work_ensure(old_gate["id"], "REVIEW", "Inspect v1", "Old reviewed contract", "RESULT_INSPECTOR", worker_id, session_id)
    claimed = lab.claim_work(work["id"], worker_id, session_id)
    lab.start_work(work["id"], worker_id, session_id)

    result = lab.supersede_gate_version(old_gate["id"], successor_gate["id"], worker_id, session_id, "A new reviewed version supersedes v1.")

    assert result["status"] == "SUPERSEDED"
    retired = lab.work_get(work["id"])
    assert retired["status"] == "SUPERSEDED"
    assert retired["lease_id"] is None
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT released_at,release_reason FROM lab_work_leases WHERE id=%s", (claimed["lease_id"],))
        lease = cur.fetchone()
        cur.execute("SELECT current_work_item_id,status FROM research_worker_sessions WHERE id=%s", (session_id,))
        session = cur.fetchone()
        cur.execute("SELECT availability_state FROM research_workers WHERE id=%s", (worker_id,))
        worker = cur.fetchone()
    assert lease["released_at"] is not None
    assert lease["release_reason"] == "GATE_SUPERSEDED"
    assert session["current_work_item_id"] is None
    assert session["status"] == "ACTIVE"
    assert worker["availability_state"] == "AVAILABLE"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_feature_gated_scheduler_claims_existing_work_or_records_healthy_idle():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, feature_flags={"BRANCH_AWARE_ASSIGNMENT": True})
    project_id = store.project_create(f"lab-v36-assign-{time.time_ns()}", "v3.6 assignment")["project_id"]
    planner = lab.join(None, "v36-assignment-planner", "CODEX", project_id)
    worker = lab.join(None, "v36-assignment-worker", "CHATGPT_WEB", project_id)
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Causal branch", "Which causal branch is active?")
    branch = lab.hypothesis_branch_create(project_id, objective["id"], scientific_distance="FAR")
    lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "E-v36", "v1", planner["worker"]["id"], planner["session_id"])
    work = lab.gate_work_ensure(
        gate["id"], "REVIEW", "Inspect E-v36", "Canonical existing work", "RESULT_INSPECTOR",
        planner["worker"]["id"], planner["session_id"], branch_id=branch["id"],
    )

    assigned = lab.portfolio_assign_existing(project_id, worker["worker"]["id"], worker["session_id"])
    assert assigned["status"] == "ASSIGNED"
    assert assigned["work_item"]["id"] == work["id"]
    assert assigned["work_item"]["branch_id"] == branch["id"]
    assert assigned["plan"]["version"] == 1

    idle = lab.portfolio_assign_existing(project_id, planner["worker"]["id"], planner["session_id"])
    assert idle["status"] == "IDLE"
    assert idle["idle_reason"] == "NO_EXISTING_ACTIONABLE_CANONICAL_WORK"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_planner_on_idle_requests_central_evaluation_without_creating_work():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, feature_flags={"BRANCH_AWARE_ASSIGNMENT": True, "PLANNER_ON_IDLE": True})
    project_id = store.project_create(f"lab-v36-planner-idle-{time.time_ns()}", "v3.6 planner on idle")["project_id"]
    joined = lab.join(None, f"v36-planner-idle-worker-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Planner", "Is missing work scientifically justified?")
    branch = lab.hypothesis_branch_create(project_id, objective["id"])
    lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    before = lab.work_list(project_id, None, 100)

    result = lab.portfolio_assign_existing(project_id, worker_id, session_id)

    assert result["status"] == "IDLE"
    assert result["idle_reason"] == "PLANNER_EVALUATION_REQUIRED"
    assert result["planner_action"] == "REQUEST_CENTRAL_PLANNER_EVALUATION"
    assert result["planner_candidates"]["planner_action"] == "CONSIDER_PROPOSAL"
    assert result["planner_candidates"]["undercovered_branches"][0]["branch_id"] == branch["id"]
    assert result["plan"]["rationale"]["new_work_created"] is False
    assert lab.work_list(project_id, None, 100) == before


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_objective_global_dependency_blocks_assignment_but_branch_local_wait_does_not():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, feature_flags={"BRANCH_AWARE_ASSIGNMENT": True})
    project_id = store.project_create(f"lab-v36-global-scope-{time.time_ns()}", "v3.6 dependency scope")["project_id"]
    worker = lab.join(None, f"v36-global-worker-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = worker["worker"]["id"], worker["session_id"]
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Scope", "Which dependency scope is blocking?")
    first_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    second_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "global-ready", "v1", worker_id, session_id)
    ready = lab.gate_work_ensure(gate["id"], "REVIEW", "Independent", "Should run absent global block.", "RESULT_INSPECTOR", worker_id, session_id, branch_id=second_branch["id"])
    local_pending = store.object_create(project_id, "ExperimentRun", {"label": "local"}, "EXPERIMENT_STARTED", "running")
    local_wait = lab.create_work(
        project_id, "VALIDATION", "Local prerequisite", "Must not block an independent branch.", "VALIDATOR", worker_id,
        created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=first_branch["id"],
        dependency_scope="BRANCH", dependencies=[{"target_type": "EXPERIMENT_RUN", "target_id": local_pending["id"], "required_statuses": ["completed"]}],
    )
    pending = store.object_create(project_id, "ExperimentRun", {"label": "global"}, "EXPERIMENT_STARTED", "running")
    global_wait = lab.create_work(
        project_id, "VALIDATION", "Global prerequisite", "Blocks this objective only.", "VALIDATOR", worker_id,
        created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=first_branch["id"],
        dependency_scope="OBJECTIVE_GLOBAL", dependencies=[{"target_type": "EXPERIMENT_RUN", "target_id": pending["id"], "required_statuses": ["completed"]}],
    )
    assert global_wait["status"] == "WAITING_DEPENDENCY"
    blocked = lab.portfolio_assign_existing(project_id, worker_id, session_id)
    assert blocked["status"] == "IDLE"
    assert blocked["idle_reason"] == "OBJECTIVE_GLOBAL_DEPENDENCY_BLOCK"
    assert blocked["global_blocks"][0]["id"] == global_wait["id"]
    assert lab.work_planner_candidates(project_id)["planner_action"] == "IDLE_OBJECTIVE_GLOBAL_BLOCK"
    store.object_update(pending["id"], {}, "completed", "EXPERIMENT_COMPLETED")
    assert lab.resolve_dependencies(project_id)["ready"] == 1
    assert lab.work_get(local_wait["id"])["status"] == "WAITING_DEPENDENCY"
    assert lab.portfolio_assign_existing(project_id, worker_id, session_id)["work_item"]["id"] == ready["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_worker_affinity_prefers_context_without_becoming_exclusive():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, feature_flags={"BRANCH_AWARE_ASSIGNMENT": True})
    project_id = store.project_create(f"lab-v36-affinity-{time.time_ns()}", "v3.6 affinity")["project_id"]
    first = lab.join(None, f"v36-affinity-first-{time.time_ns()}", "CODEX", project_id)
    preferred = lab.join(None, f"v36-affinity-preferred-{time.time_ns()}", "CODEX", project_id)
    first_id, first_session = first["worker"]["id"], first["session_id"]
    preferred_id, preferred_session = preferred["worker"]["id"], preferred["session_id"]
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Affinity", "Who can inspect this output?")
    preferred_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    other_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    preferred_gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "affinity-preferred", "v1", first_id, first_session)
    other_gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "affinity-other", "v1", first_id, first_session)
    preferred_work = lab.create_work(
        project_id, "REVIEW", "Preferred", "Prior context exists.", "RESULT_INSPECTOR", first_id,
        created_session_id=first_session, gate_id=preferred_gate["id"], authority_key=preferred_gate["authority_key"],
        canonical_subject_version="v1", authority_status="AUTHORITATIVE", subject_id="affinity-preferred",
        branch_id=preferred_branch["id"], preferred_worker_id=preferred_id, affinity_reason="Prior execution context.",
    )
    other_work = lab.create_work(
        project_id, "REVIEW", "Other", "Independent work.", "RESULT_INSPECTOR", first_id,
        created_session_id=first_session, gate_id=other_gate["id"], authority_key=other_gate["authority_key"],
        canonical_subject_version="v1", authority_status="AUTHORITATIVE", subject_id="affinity-other", branch_id=other_branch["id"],
    )
    assert lab.portfolio_assign_existing(project_id, preferred_id, preferred_session)["work_item"]["id"] == preferred_work["id"]
    assert lab.portfolio_assign_existing(project_id, first_id, first_session)["work_item"]["id"] == other_work["id"]
    lab.start_work(other_work["id"], first_id, first_session)
    lab.complete_work(other_work["id"], first_id, first_session, "Done")
    fallback_gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "affinity-fallback", "v1", first_id, first_session)
    fallback_work = lab.create_work(
        project_id, "REVIEW", "Fallback", "Context preferred but not required.", "RESULT_INSPECTOR", first_id,
        created_session_id=first_session, gate_id=fallback_gate["id"], authority_key=fallback_gate["authority_key"],
        canonical_subject_version="v1", authority_status="AUTHORITATIVE", subject_id="affinity-fallback",
        branch_id=other_branch["id"], preferred_worker_id=preferred_id, affinity_reason="Preferred worker is busy.",
    )
    assert lab.portfolio_assign_existing(project_id, first_id, first_session)["work_item"]["id"] == fallback_work["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_speculation_classes_and_expensive_budget_are_enforced():
    store = ResearchStore(TEST_DATABASE_URL)
    disabled = LabController(store)
    project_id = store.project_create(f"lab-v36-speculation-{time.time_ns()}", "v3.6 speculation")["project_id"]
    joined = disabled.join(None, "v36-speculation-worker", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = disabled.canonical_objective_create(project_id, "SCIENTIFIC", "Speculation", "Which bounded preparation is justified?")
    branch = disabled.hypothesis_branch_create(project_id, objective["id"])
    safe = disabled.create_work(
        project_id, "ANALYSIS", "Safe metric audit", "Outcome-independent inspection", "RESULT_INSPECTOR",
        worker_id, created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=branch["id"],
        speculation_class="SAFE_SPECULATIVE",
    )
    assert safe["speculation_class"] == "SAFE_SPECULATIVE"
    with pytest.raises(GPUError, match="LAB_WORK_SPECULATION_CONDITION_REQUIRED"):
        disabled.create_work(
            project_id, "ANALYSIS", "Bad conditional", "No trigger", "RESULT_INSPECTOR",
            worker_id, created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=branch["id"],
            speculation_class="CONDITIONAL_SPECULATIVE",
        )
    with pytest.raises(GPUError, match="LAB_SPECULATIVE_WORK_POLICY_DISABLED"):
        disabled.create_work(
            project_id, "RUN_EXPERIMENT", "Gated expensive run", "Not approved", "EXPERIMENT_OWNER",
            worker_id, created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=branch["id"],
            speculation_class="EXPENSIVE_SPECULATIVE", related_refs={"brain_approval": "B1"}, expected_value=1,
        )

    enabled = LabController(store, feature_flags={"SPECULATIVE_WORK_POLICY": True})
    enabled.budget_set(project_id, worker_id, session_id, {"max_concurrent_expensive_speculative_runs": 1})
    first = enabled.create_work(
        project_id, "RUN_EXPERIMENT", "Approved expensive run", "Bounded approved run", "EXPERIMENT_OWNER",
        worker_id, created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=branch["id"],
        speculation_class="EXPENSIVE_SPECULATIVE", related_refs={"brain_approval": "B1"}, expected_value=1,
    )
    enabled.claim_work(first["id"], worker_id, session_id)
    second = enabled.create_work(
        project_id, "RUN_EXPERIMENT", "Second approved run", "Must wait for budget", "EXPERIMENT_OWNER",
        worker_id, created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=branch["id"],
        speculation_class="EXPENSIVE_SPECULATIVE", related_refs={"brain_approval": "B2"}, expected_value=1,
    )
    with pytest.raises(GPUError, match="LAB_SPECULATIVE_BUDGET_EXCEEDED"):
        enabled.claim_work(second["id"], worker_id, session_id)


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_production_audit_and_cockpit_portfolio_are_read_only():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    project_id = store.project_create(f"lab-v36-audit-{time.time_ns()}", "v3.6 audit")["project_id"]
    joined = lab.join(None, f"v36-audit-worker-{time.time_ns()}", "CODEX", project_id)
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Audit", "What coordination capacity is useful?")
    branch = lab.hypothesis_branch_create(project_id, objective["id"], mechanistic_niche_id="strong-null")
    work = lab.create_work(
        project_id, "ANALYSIS", "Ready branch analysis", "Existing canonical work", "RESULT_INSPECTOR",
        joined["worker"]["id"], created_session_id=joined["session_id"],
        canonical_objective_id=objective["id"], branch_id=branch["id"],
    )
    before = lab.work_get(work["id"])
    audit = lab.portfolio_production_audit(project_id)
    state = cockpit.state_get(project_id)
    assert audit["mutated"] is False
    assert audit["existing_ready_canonical_work"] == []  # supporting work is never scheduler authority
    assert state["portfolio_scheduler"]["branch_coverage"][0]["branch_id"] == branch["id"]
    assert lab.work_get(work["id"])["work_version"] == before["work_version"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_shadow_and_audit_only_expose_workers_joined_to_the_project():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    first_project = store.project_create(f"lab-v36-worker-scope-a-{time.time_ns()}", "first scope")["project_id"]
    second_project = store.project_create(f"lab-v36-worker-scope-b-{time.time_ns()}", "second scope")["project_id"]
    first = lab.join(None, f"v36-scope-first-{time.time_ns()}", "CODEX", first_project)
    second = lab.join(None, f"v36-scope-second-{time.time_ns()}", "CODEX", second_project)
    shadow = lab.portfolio_scheduler_shadow(first_project)
    audit = lab.portfolio_production_audit(first_project)
    assert [entry["worker_id"] for entry in shadow["assignments"]] == [first["worker"]["id"]]
    assert [entry["id"] for entry in audit["workers"]["all"]] == [first["worker"]["id"]]
    assert second["worker"]["id"] not in [entry["id"] for entry in audit["workers"]["all"]]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_agenda_coverage_exposes_unbranched_active_agenda_item():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v36-agenda-uncovered-{time.time_ns()}", "v3.6 agenda coverage")["project_id"]
    agenda_item = store.object_create(
        project_id, "AgendaItem", {"question": "Which independent mechanism remains untested?"},
        "AGENDA_ITEM_CREATED", "OPEN",
    )
    coverage = lab.agenda_coverage_get(project_id)
    entry = next(item for item in coverage if item.get("agenda_item_id") == agenda_item["id"])
    assert entry["coverage_state"] == "UNCOVERED_ACTIONABLE"
    assert entry["planner_action"] == "CONSIDER_BRANCH_PROPOSAL"
    assert entry["branches"] == []


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_planner_on_idle_receives_unbranched_agenda_item_without_creating_work():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, feature_flags={"BRANCH_AWARE_ASSIGNMENT": True, "PLANNER_ON_IDLE": True})
    project_id = store.project_create(f"lab-v36-agenda-planner-{time.time_ns()}", "v3.6 agenda planner")["project_id"]
    joined = lab.join(None, f"v36-agenda-planner-worker-{time.time_ns()}", "CODEX", project_id)
    agenda_item = store.object_create(project_id, "AgendaItem", {"question": "Which strong null remains untested?"}, "AGENDA_ITEM_CREATED", "OPEN")
    result = lab.portfolio_assign_existing(project_id, joined["worker"]["id"], joined["session_id"])
    assert result["idle_reason"] == "PLANNER_EVALUATION_REQUIRED"
    assert result["planner_candidates"]["unbranched_agenda_items"][0]["agenda_item_id"] == agenda_item["id"]
    assert result["plan"]["rationale"]["unbranched_agenda_item_ids"] == [agenda_item["id"]]
    assert lab.work_list(project_id, None, 100) == []


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_historical_replay_is_read_only_and_never_claims_counterfactual_science():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v36-replay-{time.time_ns()}", "v3.6 replay")["project_id"]
    joined = lab.join(None, f"v36-replay-worker-{time.time_ns()}", "CODEX", project_id)
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Replay", "Which work was actionable?")
    branch = lab.hypothesis_branch_create(project_id, objective["id"])
    before_events = len(store.events(project_id, 500))
    replay = lab.portfolio_historical_replay(project_id)
    after_events = len(store.events(project_id, 500))
    assert replay["mutated"] is False
    assert replay["events_examined"] >= before_events
    assert replay["limitations"]["counterfactual_claim"].startswith("No suggested assignment")
    assert after_events == before_events
    assert replay["summary"]["observed_available_workers_at_end"] == [joined["worker"]["id"]]
    assert branch["id"] in {point.get("branch_id") for point in replay["schedule_points"]}


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_historical_replay_detects_waiting_release_with_existing_independent_work():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v36-replay-release-{time.time_ns()}", "v3.6 replay release")["project_id"]
    joined = lab.join(None, f"v36-replay-release-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Replay release", "Which branch remains actionable?")
    waiting_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    independent_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    waiting_gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "replay-wait", "v1", worker_id, session_id)
    ready_gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "replay-ready", "v1", worker_id, session_id)
    waiting_work = lab.gate_work_ensure(waiting_gate["id"], "REVIEW", "Wait", "Waiting branch", "RESULT_INSPECTOR", worker_id, session_id, branch_id=waiting_branch["id"])
    ready_work = lab.gate_work_ensure(ready_gate["id"], "REVIEW", "Ready", "Independent branch", "RESULT_INSPECTOR", worker_id, session_id, branch_id=independent_branch["id"])
    prerequisite = store.object_create(project_id, "ExperimentRun", {"label": "pending"}, "EXPERIMENT_STARTED", "running")
    lab.claim_work(waiting_work["id"], worker_id, session_id)
    lab.block_work(waiting_work["id"], worker_id, session_id, [{"target_type": "EXPERIMENT_RUN", "target_id": prerequisite["id"], "required_statuses": ["completed"]}])

    replay = lab.portfolio_historical_replay(project_id)
    released = [point for point in replay["schedule_points"] if point["kind"] == "WORK_RELEASED"]
    assert replay["summary"]["waiting_releases_with_independent_ready_opportunity"] == 1
    assert ready_work["id"] in released[-1]["independent_ready_work_item_ids"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_browser_runtime_marks_first_successful_connection_attached():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    project = store.project_create(f"cockpit-runtime-{time.time_ns()}", "Runtime attachment")
    joined = lab.join(None, "browser-worker", "CHATGPT_WEB", project["project_id"])

    runtime = cockpit.runtime_attach(
        project["project_id"],
        joined["worker"]["id"],
        joined["session_id"],
        "https://chatgpt.com/c/demo",
    )
    assert runtime["attached_at"] is None

    connected = cockpit.runtime_status(runtime["id"], "READY")

    assert connected["attached_at"] is not None


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_cockpit_groups_live_workers_by_project_with_identifiers():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    first = store.project_create(f"cockpit-live-a-{time.time_ns()}", "First project")
    second = store.project_create(f"cockpit-live-b-{time.time_ns()}", "Second project")
    first_worker = lab.join(None, "live-a", "CHATGPT_WEB", first["project_id"])
    second_worker = lab.join(None, "live-b", "CODEX", second["project_id"])

    by_project = {item["project_id"]: item for item in cockpit.live_workers_by_project()}

    assert by_project[first["project_id"]]["live_worker_count"] == 1
    assert by_project[first["project_id"]]["workers"][0]["display_name"] == "live-a"
    assert by_project[first["project_id"]]["workers"][0]["worker_id"] == first_worker["worker"]["id"]
    assert by_project[second["project_id"]]["workers"][0]["worker_id"] == second_worker["worker"]["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_ready_work_wake_is_deduplicated():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    cockpit = CockpitController(store, lab)
    project_id = store.project_create(f"cockpit-wake-{time.time_ns()}", "Wake deduplication")["project_id"]
    joined = lab.join(None, "wake-worker", "CHATGPT_WEB", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    cockpit.controls_set(project_id, worker_id, session_id, autopilot_enabled=True, auto_continue_enabled=True)
    runtime = cockpit.runtime_attach(project_id, worker_id, session_id, "https://chatgpt.com/c/wake-test")
    cockpit.runtime_status(runtime["id"], "READY")
    work = lab.create_work(project_id, "REVIEW", "Ready review", "Wake once", "RESULT_INSPECTOR", worker_id, created_session_id=session_id)

    assert cockpit.wake_ready_work(project_id, [work["id"]]) == {"queued": 1}
    assert cockpit.wake_ready_work(project_id, [work["id"]]) == {"queued": 0}
    assert len(cockpit.state_get(project_id)["pending_wake_requests"]) == 1


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_atomic_claim_and_dependency_reactivation_survive_store_restart():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project = store.project_create(f"lab-atomic-{time.time_ns()}", "Atomic claims")
    project_id = project["project_id"]
    first = lab.join(None, "atomic-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "atomic-b", "CODEX", project_id)
    work = lab.create_work(project_id, "REVIEW", "One canonical task", "Must only claim once", "ADVERSARIAL_REVIEWER", first["worker"]["id"], created_session_id=first["session_id"])
    barrier = threading.Barrier(2)

    def claim(joined):
        barrier.wait()
        controller = LabController(ResearchStore(TEST_DATABASE_URL))
        try:
            return controller.claim_work(work["id"], joined["worker"]["id"], joined["session_id"])["lease_id"]
        except GPUError as error:
            return error.error_type

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, (first, second)))
    assert sum(isinstance(result, str) and result != "LAB_WORK_NOT_CLAIMABLE" for result in results) == 1
    assert results.count("LAB_WORK_NOT_CLAIMABLE") == 1

    run = store.object_create(project_id, "ExperimentRun", {"label": "simulated"}, "EXPERIMENT_STARTED", "running")
    waiting = lab.create_work(project_id, "INSPECT_RESULT", "Inspect simulated run", "Wait for result", "RESULT_INSPECTOR", first["worker"]["id"], dependencies=[{"target_type": "EXPERIMENT_RUN", "target_id": run["id"], "required_statuses": ["completed"]}], created_session_id=first["session_id"])
    assert waiting["status"] == "WAITING_DEPENDENCY"
    store.object_update(run["id"], {}, "completed", "EXPERIMENT_COMPLETED")
    assert lab.resolve_dependencies(project_id)["ready"] == 1
    assert LabController(ResearchStore(TEST_DATABASE_URL)).work_get(waiting["id"])["status"] == "READY"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_concurrent_schedulers_cannot_assign_two_work_items_to_one_session():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-v36-session-race-{time.time_ns()}", "Session assignment race")["project_id"]
    joined = lab.join(None, f"v36-session-race-worker-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    first = lab.create_work(project_id, "REVIEW", "First", "First canonical work", "RESULT_INSPECTOR", worker_id, created_session_id=session_id)
    second = lab.create_work(project_id, "REVIEW", "Second", "Second canonical work", "RESULT_INSPECTOR", worker_id, created_session_id=session_id)
    barrier = threading.Barrier(2)

    def claim(work_id):
        barrier.wait()
        controller = LabController(ResearchStore(TEST_DATABASE_URL))
        try:
            return controller.claim_work(work_id, worker_id, session_id)["id"]
        except GPUError as error:
            return error.error_type

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(claim, (first["id"], second["id"])))
    assert sum(outcome in {first["id"], second["id"]} for outcome in outcomes) == 1
    assert outcomes.count("LAB_WORKER_ALREADY_ASSIGNED") == 1
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_work_item_id FROM research_worker_sessions WHERE id=%s", (session_id,))
        session = cur.fetchone()
    assert str(session["current_work_item_id"]) in {first["id"], second["id"]}


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_worker_identity_cannot_be_assigned_through_two_project_sessions():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    first_project = store.project_create(f"lab-v36-worker-identity-a-{time.time_ns()}", "first identity scope")["project_id"]
    second_project = store.project_create(f"lab-v36-worker-identity-b-{time.time_ns()}", "second identity scope")["project_id"]
    first = lab.join(None, f"v36-identity-worker-{time.time_ns()}", "CODEX", first_project)
    second = lab.join(first["worker"]["id"], None, "CODEX", second_project)
    first_work = lab.create_work(first_project, "REVIEW", "First", "First project work", "RESULT_INSPECTOR", first["worker"]["id"], created_session_id=first["session_id"])
    second_work = lab.create_work(second_project, "REVIEW", "Second", "Second project work", "RESULT_INSPECTOR", second["worker"]["id"], created_session_id=second["session_id"])
    lab.claim_work(first_work["id"], first["worker"]["id"], first["session_id"])
    with pytest.raises(GPUError) as exc_info:
        lab.claim_work(second_work["id"], second["worker"]["id"], second["session_id"])
    assert exc_info.value.error_type == "LAB_WORKER_ALREADY_ASSIGNED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_stale_completed_session_pointer_is_cleared_before_a_new_claim_after_restart():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-stale-completed-{time.time_ns()}", "stale completed ownership")['project_id']
    joined = lab.join(None, f"stale-completed-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    completed = lab.create_work(project_id, "REVIEW", "Complete old", "terminal old work", "RESULT_INSPECTOR", worker_id, created_session_id=session_id)
    lab.claim_work(completed["id"], worker_id, session_id)
    lab.complete_work(completed["id"], worker_id, session_id)
    replacement = lab.create_work(project_id, "REVIEW", "Claim new", "new ready work", "RESULT_INSPECTOR", worker_id, created_session_id=session_id)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE research_worker_sessions SET current_work_item_id=%s,status='BUSY' WHERE id=%s", (completed["id"], session_id))

    claimed = LabController(ResearchStore(TEST_DATABASE_URL)).claim_work(replacement["id"], worker_id, session_id)
    assert claimed["id"] == replacement["id"]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_work_item_id FROM research_worker_sessions WHERE id=%s", (session_id,))
        assert str(cur.fetchone()["current_work_item_id"]) == replacement["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_released_orphaned_session_pointer_cannot_block_a_fresh_session_for_same_worker():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-orphaned-session-{time.time_ns()}", "orphaned session ownership")['project_id']
    old = lab.join(None, f"orphaned-worker-{time.time_ns()}", "CODEX", project_id)
    fresh = lab.join(old["worker"]["id"], None, "CODEX", project_id)
    worker_id = old["worker"]["id"]
    old_work = lab.create_work(project_id, "REVIEW", "Old lease", "orphaned old lease", "RESULT_INSPECTOR", worker_id, created_session_id=old["session_id"])
    lab.claim_work(old_work["id"], worker_id, old["session_id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_leases SET released_at=NOW(),release_reason='ORPHANED_SESSION' WHERE work_item_id=%s", (old_work["id"],))
        cur.execute("UPDATE research_worker_sessions SET current_work_item_id=%s,status='BUSY' WHERE id=%s", (old_work["id"], old["session_id"]))
    new_work = lab.create_work(project_id, "REVIEW", "Fresh lease", "fresh ready work", "RESULT_INSPECTOR", worker_id, created_session_id=fresh["session_id"])

    assert lab.claim_work(new_work["id"], worker_id, fresh["session_id"])["id"] == new_work["id"]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_work_item_id FROM research_worker_sessions WHERE id=%s", (old["session_id"],))
        assert cur.fetchone()["current_work_item_id"] is None


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_sync_cleans_lost_terminal_ownership_projection():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-sync-stale-{time.time_ns()}", "sync stale ownership")['project_id']
    joined = lab.join(None, f"sync-stale-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    old_work = lab.create_work(project_id, "REVIEW", "Complete sync old", "terminal sync work", "RESULT_INSPECTOR", worker_id, created_session_id=session_id)
    lab.claim_work(old_work["id"], worker_id, session_id)
    lab.complete_work(old_work["id"], worker_id, session_id)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE research_worker_sessions SET current_work_item_id=%s,status='BUSY' WHERE id=%s", (old_work["id"], session_id))

    synced = lab.sync(session_id, project_id, current_work_item_id=old_work["id"])
    assert synced["lease_state"] == "LEASE_LOST"
    assert synced["sync_status"] == "WORK_COMPLETED_ELSEWHERE"
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT current_work_item_id FROM research_worker_sessions WHERE id=%s", (session_id,))
        assert cur.fetchone()["current_work_item_id"] is None


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_live_lease_ownership_still_blocks_competing_session_claims():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-live-owner-{time.time_ns()}", "live owner remains exclusive")['project_id']
    owner = lab.join(None, f"live-owner-{time.time_ns()}", "CODEX", project_id)
    competitor = lab.join(owner["worker"]["id"], None, "CODEX", project_id)
    worker_id = owner["worker"]["id"]
    owned = lab.create_work(project_id, "REVIEW", "Live owned", "must retain ownership", "RESULT_INSPECTOR", worker_id, created_session_id=owner["session_id"])
    contender = lab.create_work(project_id, "REVIEW", "Competing", "must not claim", "RESULT_INSPECTOR", worker_id, created_session_id=competitor["session_id"])
    lab.claim_work(owned["id"], worker_id, owner["session_id"])

    with pytest.raises(GPUError) as exc_info:
        lab.claim_work(contender["id"], worker_id, competitor["session_id"])
    assert exc_info.value.error_type == "LAB_WORKER_ALREADY_ASSIGNED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_controller_restart_reconstructs_waiting_branch_and_reclaims_after_dependency():
    store = ResearchStore(TEST_DATABASE_URL)
    flags = {"WAITING_WORK_RELEASE": True, "BRANCH_AWARE_ASSIGNMENT": True}
    lab = LabController(store, feature_flags=flags)
    project_id = store.project_create(f"lab-v36-restart-{time.time_ns()}", "v3.6 restart durability")["project_id"]
    joined = lab.join(None, f"v36-restart-worker-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Restart", "Does waiting work survive restart?")
    branch = lab.hypothesis_branch_create(project_id, objective["id"])
    lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "restart-run", "v1", worker_id, session_id)
    work = lab.gate_work_ensure(gate["id"], "REVIEW", "Inspect after restart", "Wait for canonical run.", "RESULT_INSPECTOR", worker_id, session_id, branch_id=branch["id"])
    run = store.object_create(project_id, "ExperimentRun", {"label": "pending"}, "EXPERIMENT_STARTED", "running")
    lab.claim_work(work["id"], worker_id, session_id)
    lab.block_work(work["id"], worker_id, session_id, [{"target_type": "EXPERIMENT_RUN", "target_id": run["id"], "required_statuses": ["completed"]}])

    restarted = LabController(ResearchStore(TEST_DATABASE_URL), feature_flags=flags)
    coverage = restarted.branch_coverage_get(project_id)
    assert coverage[0]["waiting_work_item_ids"] == [work["id"]]
    assert coverage[0]["active_worker_count"] == 0
    store.object_update(run["id"], {}, "completed", "EXPERIMENT_COMPLETED")
    assert restarted.resolve_dependencies(project_id)["ready"] == 1
    assigned = restarted.portfolio_assign_existing(project_id, worker_id, session_id)
    assert assigned["work_item"]["id"] == work["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_unsatisfied_dependency_demotes_ready_work_and_blocks_claim():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-dependency-invariant-{time.time_ns()}", "Dependency invariant")["project_id"]
    worker = lab.join(None, "dependency-worker", "CODEX", project_id)
    prerequisite = lab.create_work(
        project_id, "ENGINEERING", "Recover service", "Repair upstream service", "ENGINEER",
        worker["worker"]["id"], created_session_id=worker["session_id"],
    )
    dependent = lab.create_work(
        project_id, "EXPERIMENT_DESIGN", "Preregister", "Must wait", "SCIENTIST",
        worker["worker"]["id"], created_session_id=worker["session_id"],
        dependencies=[{"target_type": "WORK_ITEM", "target_id": prerequisite["id"], "required_statuses": ["COMPLETED"]}],
    )
    assert dependent["status"] == "WAITING_DEPENDENCY"
    # Simulate a legacy/partial write that attached a dependency to READY work.
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_items SET status='READY',blocked_reason=NULL WHERE id=%s", (dependent["id"],))
    assert lab.resolve_dependencies(project_id)["waiting"] == 1
    assert lab.work_get(dependent["id"])["status"] == "WAITING_DEPENDENCY"
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_items SET status='READY',blocked_reason=NULL WHERE id=%s", (dependent["id"],))
    with pytest.raises(GPUError) as exc:
        lab.claim_work(dependent["id"], worker["worker"]["id"], worker["session_id"])
    assert exc.value.error_type == "LAB_WORK_DEPENDENCY_UNSATISFIED"
    assert lab.work_get(dependent["id"])["status"] == "WAITING_DEPENDENCY"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_dormant_equivalence_is_reserved_before_it_can_wake():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-equivalence-reconcile-{time.time_ns()}", "Equivalent work") ["project_id"]
    worker = lab.join(None, "equivalence-worker", "CODEX", project_id)
    prerequisite = lab.create_work(
        project_id, "ENGINEERING", "Implement", "Complete first", "ENGINEER",
        worker["worker"]["id"], created_session_id=worker["session_id"],
    )
    dependency = [{"target_type": "WORK_ITEM", "target_id": prerequisite["id"], "required_statuses": ["COMPLETED"]}]
    first = lab.create_work(
        project_id, "REVIEW", "Canonical review", "One review", "ADVERSARIAL_REVIEWER",
        worker["worker"]["id"], created_session_id=worker["session_id"], dependencies=dependency,
        equivalence_key="same-review", dormant_until_dependencies=True,
    )
    with pytest.raises(GPUError, match="LAB_EQUIVALENT_WORK_ACTIVE"):
        lab.create_work(
            project_id, "REVIEW", "Duplicate review", "Same review", "ADVERSARIAL_REVIEWER",
            worker["worker"]["id"], created_session_id=worker["session_id"], dependencies=dependency,
            equivalence_key="same-review", dormant_until_dependencies=True,
        )
    claimed = lab.claim_work(prerequisite["id"], worker["worker"]["id"], worker["session_id"])
    lab.complete_work(claimed["id"], worker["worker"]["id"], worker["session_id"], summary="Implemented")
    assert lab.resolve_dependencies(project_id) == {"ready": 0, "waiting": 0, "invalidated": 0}
    assert lab.work_get(first["id"])["status"] == "READY"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_lab_state_summary_does_not_return_historical_large_object_data():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-compact-state-{time.time_ns()}", "Compact state") ["project_id"]
    store.object_create(project_id, "Artifact", {"content": "x" * 200_000}, "ARTIFACT_CREATED", "COMPLETED")
    summary = lab.state_get(project_id)
    assert summary["research_state_version"] == 1
    assert "content" not in str(summary)
    assert len(str(summary)) < 20_000

@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_budget_and_expired_lease_release_work_without_touching_experiment():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project = store.project_create(f"lab-budget-{time.time_ns()}", "Budget and lease")
    project_id = project["project_id"]
    first = lab.join(None, "budget-a", "CHATGPT_WEB", project_id)
    second = lab.join(None, "budget-b", "CODEX", project_id)
    lab.budget_set(project_id, first["worker"]["id"], first["session_id"], {"max_active_workers": 1})
    first_work = lab.create_work(project_id, "REVIEW", "First", "First", "ADVERSARIAL_REVIEWER", first["worker"]["id"], created_session_id=first["session_id"])
    second_work = lab.create_work(project_id, "REVIEW", "Second", "Second", "ADVERSARIAL_REVIEWER", second["worker"]["id"], created_session_id=second["session_id"])
    lab.claim_work(first_work["id"], first["worker"]["id"], first["session_id"])
    with pytest.raises(GPUError, match="LAB_WORKER_BUDGET_EXCEEDED"):
        lab.claim_work(second_work["id"], second["worker"]["id"], second["session_id"])

    # Lease recovery releases only coordination ownership; the external run remains canonical and running.
    run = store.object_create(project_id, "ExperimentRun", {"label": "must-not-cancel"}, "EXPERIMENT_STARTED", "running")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE lab_work_leases SET expires_at=NOW() - INTERVAL '1 second' WHERE work_item_id=%s", (first_work["id"],))
    assert lab.recover_stale_leases(project_id)["recovered"] == 1
    assert lab.work_get(first_work["id"])["status"] == "READY"
    assert store.object_get(run["id"])["status"] == "running"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_orphaned_running_work_without_a_lease_is_recovered():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project_id = store.project_create(f"lab-orphan-{time.time_ns()}", "Orphan recovery")["project_id"]
    joined = lab.join(None, "orphaned-worker", "CODEX", project_id)
    work = lab.create_work(
        project_id, "REVIEW", "Recover me", "No owner may run forever", "REVIEWER",
        joined["worker"]["id"], created_session_id=joined["session_id"],
    )
    lab.claim_work(work["id"], joined["worker"]["id"], joined["session_id"])
    lab.start_work(work["id"], joined["worker"]["id"], joined["session_id"])
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM lab_work_leases WHERE work_item_id=%s", (work["id"],))
        cur.execute(
            "UPDATE research_worker_sessions SET status='EXPIRED',last_heartbeat_at="
            "NOW() - INTERVAL '1 hour' WHERE id=%s",
            (joined["session_id"],),
        )

    assert lab.recover_stale_leases(project_id) == {"recovered": 1}
    assert lab.work_get(work["id"])["status"] == "READY"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_live_lease_is_never_orphan_reclaimed_for_an_active_session():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project_id = store.project_create(f"lab-live-lease-{time.time_ns()}", "Lease authority") ["project_id"]
    joined = lab.join(None, "live-lease-worker", "CODEX", project_id)
    work = lab.create_work(
        project_id, "REVIEW", "Keep ownership", "Live lease is authoritative", "REVIEWER",
        joined["worker"]["id"], created_session_id=joined["session_id"],
    )
    claimed = lab.claim_work(work["id"], joined["worker"]["id"], joined["session_id"])
    lab.start_work(claimed["id"], joined["worker"]["id"], joined["session_id"])
    # Reproduce the formerly unsafe projection: session remains ACTIVE but its
    # heartbeat looks stale.  Its unexpired lease must still win.
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE research_worker_sessions SET status='ACTIVE',last_heartbeat_at="
            "NOW() - INTERVAL '1 hour' WHERE id=%s",
            (joined["session_id"],),
        )

    recovered = lab.recover_stale_leases(project_id)

    assert recovered["recovered"] == 0
    current = lab.work_get(work["id"])
    assert current["status"] == "RUNNING"
    assert current["assigned_session_id"] == joined["session_id"]
    assert current["lease_id"] == claimed["lease_id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_start_work_explains_an_unknown_work_item_id():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-missing-id-{time.time_ns()}", "ID diagnostics")["project_id"]
    joined = lab.join(None, "id-diagnostic-worker", "CODEX", project_id)

    missing_id = str(uuid.uuid4())
    with pytest.raises(GPUError) as exc_info:
        lab.start_work(missing_id, joined["worker"]["id"], joined["session_id"])

    assert exc_info.value.error_type == "LAB_WORK_NOT_FOUND"
    assert exc_info.value.message == f"Expected a Lab WorkItem ID; no WorkItem exists for {missing_id}"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_attached_execution_cannot_return_to_ready_after_worker_disconnect():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project_id = store.project_create(f"lab-attachment-{time.time_ns()}", "Attached execution")["project_id"]
    joined = lab.join(None, "attached-worker", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    run = store.object_create(project_id, "ExperimentRun", {"label": "canonical"}, "EXPERIMENT_STARTED", "running")
    work = lab.create_work(project_id, "TRAINING_RUN", "Run canonical", "Launch once", "EXECUTION", worker_id, created_session_id=session_id)
    lab.claim_work(work["id"], worker_id, session_id)
    attached = lab.attach_experiment_run(work["id"], worker_id, session_id, run["id"])
    assert attached["status"] == "RUNNING_DETACHED"
    assert attached["related_refs"]["experiment_run_id"] == run["id"]
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT availability_state FROM research_workers WHERE id=%s", (worker_id,))
        worker = cur.fetchone()
    assert worker["availability_state"] == "AVAILABLE"
    assert lab.work_list(project_id, ["READY"]) == []

    assert lab.experiment_run_terminal(run["id"], "completed") == {"result_ready": 1}
    assert lab.work_get(work["id"])["status"] == "RESULT_READY"
    assert lab.work_list(project_id, ["READY"]) == []
    store.object_update(run["id"], {"inspection": {"mode": "fixture"}}, "RESULT_INSPECTED")
    assert lab.experiment_run_inspected(run["id"]) == {"completed": 1}
    assert lab.work_get(work["id"])["status"] == "COMPLETED"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_v36_detached_gpu_run_releases_worker_for_existing_independent_work():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, feature_flags={"BRANCH_AWARE_ASSIGNMENT": True})
    project_id = store.project_create(f"lab-v36-detach-reassign-{time.time_ns()}", "v3.6 GPU detach")["project_id"]
    joined = lab.join(None, f"v36-detach-worker-{time.time_ns()}", "CODEX", project_id)
    worker_id, session_id = joined["worker"]["id"], joined["session_id"]
    objective = lab.canonical_objective_create(project_id, "SCIENTIFIC", "Detach", "Can long execution release reasoning capacity?")
    run_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    inspect_branch = lab.hypothesis_branch_create(project_id, objective["id"])
    lab.hypothesis_portfolio_ensure(project_id, objective["id"])
    run_work = lab.create_work(project_id, "TRAINING_RUN", "Launch", "Canonical long run.", "EXECUTION", worker_id, created_session_id=session_id, canonical_objective_id=objective["id"], branch_id=run_branch["id"])
    gate = lab.gate_ensure(project_id, "RESULT_INSPECTION", "detach-independent", "v1", worker_id, session_id)
    ready_work = lab.gate_work_ensure(gate["id"], "REVIEW", "Inspect", "Existing independent review.", "RESULT_INSPECTOR", worker_id, session_id, branch_id=inspect_branch["id"])
    run = store.object_create(project_id, "ExperimentRun", {"label": "long"}, "EXPERIMENT_STARTED", "running")
    lab.claim_work(run_work["id"], worker_id, session_id)
    assert lab.attach_experiment_run(run_work["id"], worker_id, session_id, run["id"])["status"] == "RUNNING_DETACHED"
    assert lab.portfolio_assign_existing(project_id, worker_id, session_id)["work_item"]["id"] == ready_work["id"]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_scientific_gate_authority_preflight_unlock_and_supersession():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store, lease_seconds=30)
    project_id = store.project_create(f"lab-gates-{time.time_ns()}", "Gate coordination")["project_id"]
    worker = lab.join(None, "gate-worker", "CODEX", project_id)
    worker_id, session_id = worker["worker"]["id"], worker["session_id"]

    gate = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment-v1", "v1", worker_id, session_id)
    assert lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment-v1", "v1", worker_id, session_id)["id"] == gate["id"]
    first = lab.gate_work_ensure(gate["id"], "REVIEW", "Canonical review", "Review E1", "RESULT_INSPECTOR", worker_id, session_id)
    second = lab.gate_work_ensure(gate["id"], "REVIEW", "Duplicate review", "Must reuse", "RESULT_INSPECTOR", worker_id, session_id)
    assert first["id"] == second["id"]
    assert first["authority_status"] == "AUTHORITATIVE"

    dependent = lab.create_work(
        project_id, "GENERALIZATION", "Conditional next work", "Wait for gate", "SCIENTIST",
        worker_id, dependencies=[{"target_type": "SCIENTIFIC_GATE", "target_id": gate["id"], "required_statuses": ["PASS"]}],
        created_session_id=session_id, dormant_until_dependencies=True,
    )
    assert dependent["status"] == "DORMANT"
    failed = lab.preflight_run(gate["id"], worker_id, session_id, {"checkpoint": False, "tokenizer": True})
    assert failed["preflight"]["status"] == "FAIL"
    with pytest.raises(GPUError, match="PREFLIGHT_NOT_PASS"):
        lab.gate_resolve(gate["id"], worker_id, session_id, "PASS")

    passed = lab.preflight_run(gate["id"], worker_id, session_id, {"checkpoint": True, "tokenizer": True})
    assert passed["preflight"]["status"] == "PASS"
    with pytest.raises(GPUError, match="SEMANTIC_REVIEW_REQUIRED"):
        lab.gate_resolve(gate["id"], worker_id, session_id, "PASS")
    claimed_review = lab.claim_work(first["id"], worker_id, session_id)
    lab.complete_work(claimed_review["id"], worker_id, session_id, summary="Semantic review passed")
    resolved = lab.gate_resolve(gate["id"], worker_id, session_id, "PASS", first["id"], rationale="Semantic review passed")
    assert resolved["gate"]["status"] == "PASS"
    assert lab.work_get(dependent["id"])["status"] == "READY"
    repeated = lab.gate_resolve(gate["id"], worker_id, session_id, "PASS", first["id"])
    assert repeated["idempotent"] is True
    with pytest.raises(GPUError, match="ALREADY_RESOLVED"):
        lab.gate_resolve(gate["id"], worker_id, session_id, "FAIL", first["id"])

    obsolete = lab.create_work(
        project_id, "REVIEW", "Obsolete E1 follow-up", "Bound to E1", "RESULT_INSPECTOR",
        worker_id, related_refs={"experiment_id": "experiment-v1"}, created_session_id=session_id,
    )
    claimed = lab.claim_work(obsolete["id"], worker_id, session_id)
    successor = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment-v2", "v2", worker_id, session_id)
    summary = lab.supersede_subject(project_id, "experiment-v1", "experiment-v2", "Corrected canonical experiment", worker_id, session_id, successor["id"])
    assert summary["work_items_superseded"] == 1
    assert lab.work_get(first["id"])["status"] == "COMPLETED"
    assert lab.work_get(obsolete["id"])["status"] == "SUPERSEDED"
    synced = lab.sync(session_id, project_id, current_work_item_id=claimed["id"], expected_work_version=claimed["work_version"])
    assert synced["lease_state"] == "LEASE_LOST"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_concurrent_gate_authority_creation_reuses_one_work_item():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    project_id = store.project_create(f"lab-gate-concurrent-{time.time_ns()}", "Concurrent gate authority")["project_id"]
    first = lab.join(None, "concurrent-gate-a", "CODEX", project_id)
    second = lab.join(None, "concurrent-gate-b", "LOCAL_AGENT", project_id)
    gate = lab.gate_ensure(project_id, "RESULT_ASSESSMENT", "experiment", "v1", first["worker"]["id"], first["session_id"])
    barrier = threading.Barrier(2)

    def ensure(joined):
        barrier.wait()
        controller = LabController(ResearchStore(TEST_DATABASE_URL))
        return controller.gate_work_ensure(
            gate["id"], "REVIEW", "Canonical review", "One authority", "RESULT_INSPECTOR",
            joined["worker"]["id"], joined["session_id"],
        )["id"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(ensure, (first, second)))
    assert result[0] == result[1]
    authorities = lab.work_list(project_id, list(ACTIVE_WORK_STATUSES))
    assert [item["id"] for item in authorities if item["authority_status"] == "AUTHORITATIVE"] == [result[0]]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_project_scope_and_message_acknowledgement_require_active_session():
    store = ResearchStore(TEST_DATABASE_URL)
    lab = LabController(store)
    one = store.project_create(f"lab-scope-one-{time.time_ns()}", "One")["project_id"]
    two = store.project_create(f"lab-scope-two-{time.time_ns()}", "Two")["project_id"]
    sender = lab.join(None, "scope-sender", "CODEX", one)
    recipient = lab.join(None, "scope-recipient", "CHATGPT_WEB", one)
    outsider = lab.join(None, "scope-outsider", "OTHER", two)
    message = lab.message_send(one, sender["worker"]["id"], sender["session_id"], "REQUEST_REVIEW", "Scope", "Project one only", to_worker_id=recipient["worker"]["id"])
    assert lab.message_list(two, outsider["worker"]["id"]) == []
    assert lab.message_mark_read(one, recipient["worker"]["id"], recipient["session_id"], [message["id"]]) == {"marked_read": 1}
    assert lab.message_list(one, recipient["worker"]["id"], unread_only=True) == []
    with pytest.raises(GPUError, match="LAB_MESSAGE_RECIPIENT_NOT_IN_PROJECT"):
        lab.message_send(one, sender["worker"]["id"], sender["session_id"], "REQUEST_REVIEW", "Bad", "No cross-project recipient", to_worker_id=outsider["worker"]["id"])
