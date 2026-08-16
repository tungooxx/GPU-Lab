from __future__ import annotations

import httpx
import pytest

from gpu_lab.errors import GPUError
from gpu_lab.research_operators import (
    HttpResearchOperatorProvider,
    ResearchOperatorService,
)


class OperatorStore:
    def __init__(self):
        self.items = [
            {
                "id": "agenda-1",
                "project_id": "project-1",
                "kind": "AgendaItem",
                "status": "ACTIVE",
                "data": {"question": "Does anchor state carry viewpoint evidence?"},
            },
            {
                "id": "niche-1",
                "project_id": "project-1",
                "kind": "HypothesisNiche",
                "status": "ACTIVE",
                "data": {"name": "state propagation"},
            },
            {
                "id": "dead-1",
                "project_id": "project-1",
                "kind": "NegativeResult",
                "status": "REFUTED",
                "data": {"failed_assumption": "output correlation is causal"},
            },
        ]

    def object_get(self, object_id):
        return next(item for item in self.items if item["id"] == object_id)

    def objects_list(self, project_id, kind, statuses=None, limit=25):
        rows = [
            item
            for item in self.items
            if item["project_id"] == project_id
            and item["kind"] == kind
            and (not statuses or item["status"] in statuses)
        ]
        return rows[:limit]


class QDStub:
    def __init__(self):
        self.drafts = []

    def screen(self, project_id, draft):
        self.drafts.append((project_id, draft))
        return {"accepted": True, "similar_dead_hypothesis_ids": ["dead-1"]}

    def create(self, project_id, draft):
        return {"id": f"created-{len(self.drafts)}", "project_id": project_id, "data": draft}


class OperatorProvider:
    name = "fixture"
    model = "fixture-model"
    model_version = "fixture-version"

    def __init__(self, hypotheses=3):
        self.hypotheses = hypotheses
        self.calls = []

    async def run(self, operator_name, context):
        self.calls.append((operator_name, context))
        if operator_name == "NullModelCritic":
            return {
                "target_claim": context["target_claim"],
                "alternative_explanations": [
                    {
                        "name": "magnitude matched random state",
                        "mechanism": "Perturbation magnitude, not semantic state, changes output",
                        "why_plausible": "The target intervention changes activation norm",
                        "evidence_for": [],
                        "evidence_against": [],
                        "discriminating_control": "Run a magnitude-matched random substitution",
                        "estimated_cost": "LOW",
                    }
                ],
                "missing_controls": ["magnitude-matched random substitution"],
                "promotion_risk": "Mechanism is not isolated",
                "recommended_null_test": "Run the matched random substitution first",
            }
        if operator_name != "HypothesisGenerator":
            return {
                "findings": [
                    {
                        "code": "CHEAP_NULL_UNTESTED",
                        "severity": "WARNING",
                        "description": "A cheap control remains untested",
                        "related_ids": [],
                        "suggested_action": "Run the control",
                    }
                ]
            }
        return {
            "hypotheses": [
                {
                    "statement": f"Scoped candidate mechanism {index}",
                    "mechanism": f"Anchor state propagation mechanism variant {index}",
                    "state_variables": ["anchor_state", "decoder_state"],
                    "information_path": ["viewpoint", "anchor_state", "decoder_state"],
                    "assumptions": ["baseline reproduced"],
                    "inherited_assumptions": [],
                    "assumptions_removed": ["output correlation is sufficient"],
                    "scientific_difference": f"Variant {index} changes the carrier stage",
                    "niche_id": "niche-1",
                    "supporting_evidence": [],
                    "against_evidence": ["dead-1"],
                    "unique_predictions": [f"Intervention changes layer {index}"],
                    "cheapest_kill_test": f"Swap state before layer {index}",
                    "alternative_explanations": ["magnitude artifact"],
                    "expected_scope": {"architectures": ["VRCNet"]},
                    "novelty_risk": "May overlap a dead correlation lineage",
                }
                for index in range(self.hypotheses)
            ]
        }


@pytest.mark.asyncio
async def test_hypothesis_generator_is_typed_bounded_screened_and_advisory():
    qd = QDStub()
    provider = OperatorProvider()
    service = ResearchOperatorService(OperatorStore(), qd, provider)

    result = await service.generate_hypotheses("project-1", "agenda-1", persist=False)

    assert len(result["hypotheses"]) == 3
    assert len(result["screened_candidates"]) == 3
    assert all(item["persisted"] is None for item in result["screened_candidates"])
    assert all(draft["operator_provenance"]["context_hash"] for _project, draft in qd.drafts)
    assert result["warning"].startswith("Model output is advisory")
    bounded_context = provider.calls[0][1]
    assert bounded_context["negative_results"][0]["id"] == "dead-1"
    assert "database_url" not in str(bounded_context).lower()


@pytest.mark.asyncio
async def test_hypothesis_generator_rejects_unbounded_candidate_count():
    service = ResearchOperatorService(OperatorStore(), QDStub(), OperatorProvider(hypotheses=2))

    with pytest.raises(GPUError) as error:
        await service.generate_hypotheses("project-1", "agenda-1")

    assert error.value.error_type == "RESEARCH_OPERATOR_INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_null_and_design_critics_return_versioned_typed_advice():
    service = ResearchOperatorService(OperatorStore(), QDStub(), OperatorProvider())

    null = await service.null_model_critique(
        "project-1", "Anchor substitution proves semantic transport", {"metric": "CD"}
    )
    design = await service.critique(
        "ExperimentalDesignCritic", "project-1", {"control": "none"}
    )

    assert null["alternative_explanations"][0]["discriminating_control"]
    assert null["provenance"]["prompt_version"] == "brain-v2-operators-1"
    assert design["findings"][0]["code"] == "CHEAP_NULL_UNTESTED"
    assert design["provenance"]["operator_name"] == "ExperimentalDesignCritic"


@pytest.mark.asyncio
async def test_http_operator_provider_rejects_malformed_or_structured_worker_errors():
    async def handler(request):
        assert request.headers["Authorization"] == "Bearer scoped-token"
        if request.url.path.endswith("operator"):
            return httpx.Response(
                502,
                json={
                    "error": {
                        "type": "RESEARCH_OPERATOR_PROVIDER_FAILURE",
                        "message": "sanitized failure",
                        "retryable": True,
                    }
                },
            )
        return httpx.Response(200, json=[])

    provider = HttpResearchOperatorProvider(
        "http://worker:8010",
        "scoped-token",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(GPUError) as error:
        await provider.run("HypothesisGenerator", {"bounded": True})

    assert error.value.error_type == "RESEARCH_OPERATOR_PROVIDER_FAILURE"
    assert error.value.retryable is True
