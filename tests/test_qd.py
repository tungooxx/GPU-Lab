import uuid

import pytest

from gpu_lab.errors import GPUError
from gpu_lab.qd import (
    HypothesisGenerator,
    HypothesisQDService,
    ProximityCritic,
    ScientificReflector,
)


class FakeStore:
    def __init__(self):
        self.vector_available = True
        self.items = []
        self.edges = []
        self.related = []
        self.semantic = []
        self.embeddings = []

    def object_create(self, project_id, kind, data, event_type, status="ACTIVE"):
        item = {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "kind": kind,
            "status": status,
            "data": data,
            "event_type": event_type,
        }
        self.items.append(item)
        return item

    def object_get(self, object_id):
        return next(item for item in self.items if item["id"] == object_id)

    def object_update(self, object_id, data_update, status, event_type):
        item = self.object_get(object_id)
        item["data"] = {**item["data"], **data_update}
        item["status"] = status
        item["event_type"] = event_type
        return item

    def objects_list(self, project_id, kind=None, statuses=None, limit=100, data_filters=None):
        rows = [
            item
            for item in self.items
            if item["project_id"] == project_id
            and (kind is None or item["kind"] == kind)
            and (not statuses or item["status"] in statuses)
            and all(item["data"].get(key) == value for key, value in (data_filters or {}).items())
        ]
        return rows if limit is None else rows[:limit]

    def related_hypotheses(self, _project_id, _mechanism, _limit):
        return self.related

    def semantic_search(self, _project_id, _embedding, _kind, _limit):
        return self.semantic

    def edge_create(self, source, target, relation):
        self.edges.append((source, target, relation))

    def hypothesis_niche_create(self, project_id, name, description, diversity_signature):
        existing = self.objects_list(
            project_id, "HypothesisNiche", limit=None, data_filters={"name": name}
        )
        if existing:
            data = existing[0]["data"]
            if data["description"] != description or data["diversity_signature"] != diversity_signature:
                raise GPUError("HYPOTHESIS_NICHE_CONFLICT", name)
            return {**existing[0], "idempotent_replay": True}
        return self.object_create(
            project_id,
            "HypothesisNiche",
            {
                "name": name,
                "description": description,
                "active_best_hypothesis_id": None,
                "diversity_signature": diversity_signature,
            },
            "HYPOTHESIS_NICHE_CREATED",
        )

    def hypothesis_niche_set_best(self, niche_id, hypothesis_id, rationale):
        niche = self.object_get(niche_id)
        hypothesis = self.object_get(hypothesis_id)
        if hypothesis["status"] not in {"ACTIVE", "SURVIVES_INITIAL_TEST", "SUPPORTED"}:
            raise GPUError("HYPOTHESIS_NOT_ACTIVE", hypothesis["status"])
        return self.object_update(
            niche_id,
            {"active_best_hypothesis_id": hypothesis_id, "selection_rationale": rationale},
            niche["status"],
            "HYPOTHESIS_NICHE_BEST_CHANGED",
        )

    def hypothesis_create_with_edges(self, project_id, data, edges):
        item = self.object_create(
            project_id, "Hypothesis", data, "QD_HYPOTHESIS_CREATED"
        )
        self.edges.extend((source, item["id"], relation) for source, relation in edges)
        return item

    def embedding_set(self, object_id, embedding):
        self.embeddings.append((object_id, embedding))


def draft(niche_id, **updates):
    value = {
        "mechanism": "Anchor state transmits viewpoint evidence into the decoder carrier",
        "prediction": "Replacing anchor state changes the downstream carrier",
        "kill_condition": "Replacement leaves the carrier unchanged under fixed decoder state",
        "niche_id": niche_id,
        "parent_ids": [],
        "assumptions": ["anchor state is causally upstream"],
        "variables": ["anchor_state", "carrier"],
        "information_path": ["viewpoint", "anchor_state", "carrier"],
        "scope": "VRCNet frozen inference",
    }
    return {**value, **updates}


def test_qd_niches_are_typed_and_idempotent():
    store = FakeStore()
    service = HypothesisQDService(store)

    niche = service.niche_create(
        "project",
        "state propagation",
        "Mechanisms in which intermediate state transmits failure information",
        {"family": "state propagation", "stage": "decoder entry"},
    )
    replay = service.niche_create(
        "project",
        "state propagation",
        "Mechanisms in which intermediate state transmits failure information",
        {"family": "state propagation", "stage": "decoder entry"},
    )

    assert replay["id"] == niche["id"]
    assert replay["idempotent_replay"] is True
    assert service.niche_list("project") == [niche]

    with pytest.raises(GPUError) as error:
        service.niche_create(
            "project", "state propagation", "Changed definition", {"family": "other"}
        )
    assert error.value.error_type == "HYPOTHESIS_NICHE_CONFLICT"


def test_qd_blocks_unexplained_descendant_of_dead_mechanism_then_preserves_lineage():
    store = FakeStore()
    service = HypothesisQDService(store)
    niche = service.niche_create("project", "routing", "Evidence routing", {"family": "routing"})
    parent = store.object_create(
        "project",
        "Hypothesis",
        {"mechanism": "Parent mechanism", "parent_ids": []},
        "HYPOTHESIS_CREATED",
    )
    dead = store.object_create(
        "project",
        "NegativeResult",
        {
            "proposal": "Anchor state transmits viewpoint evidence into the decoder carrier",
            "assumptions": ["anchor state is causally upstream"],
        },
        "NEGATIVE_RESULT_CREATED",
    )
    store.related = [
        {
            **dead,
            "lexical_similarity": 0.9,
            "containment_similarity": 0.9,
        }
    ]

    with pytest.raises(GPUError) as error:
        service.create("project", draft(niche["id"], parent_ids=[parent["id"]]))

    assert error.value.error_type == "HYPOTHESIS_PROXIMITY_BLOCKED"

    created = service.create(
        "project",
        draft(
            niche["id"],
            parent_ids=[parent["id"]],
            scientific_difference="Uses an internal state intervention rather than static association",
        ),
    )

    assert created["data"]["ancestor_ids"] == [parent["id"]]
    assert created["data"]["similar_dead_hypothesis_ids"] == [dead["id"]]
    assert (parent["id"], created["id"], "PARENT_OF") in store.edges
    assert (niche["id"], created["id"], "CONTAINS_HYPOTHESIS") in store.edges


def test_qd_combines_vector_and_structured_comparison_without_equating_them():
    store = FakeStore()
    service = HypothesisQDService(store)
    niche = service.niche_create(
        "project", "decoder amplification", "Decoder mechanisms", {"family": "decoder"}
    )
    active = store.object_create(
        "project",
        "Hypothesis",
        {
            "mechanism": "Decoder creates the carrier",
            "niche_id": niche["id"],
            "assumptions": ["decoder state is sufficient"],
            "variables": ["decoder_state", "carrier"],
            "information_path": ["decoder_state", "carrier"],
            "scope": "VRCNet frozen inference",
        },
        "HYPOTHESIS_CREATED",
    )
    store.semantic = [{**active, "distance": 0.08}]

    result = service.screen(
        "project",
        draft(
            niche["id"],
            mechanism="Decoder state independently amplifies evidence into the carrier",
            assumptions=["decoder state is sufficient"],
            variables=["decoder_state", "carrier"],
            information_path=["decoder_state", "carrier"],
            embedding=[0.1, 0.2],
            scientific_difference="Tests sufficiency under a fixed anchor state",
        ),
    )

    match = result["matches"][0]
    assert match["semantic_similarity"] == 0.92
    assert match["structured_similarity"] > 0.8
    assert result["warning"].endswith("never scientific truth.")
    assert result["accepted"] is True


def test_qd_embedding_cache_unavailable_does_not_rollback_canonical_hypothesis():
    store = FakeStore()
    store.vector_available = False
    service = HypothesisQDService(store)
    niche = service.niche_create("project", "routing", "Routing", {"family": "routing"})

    created = service.create("project", draft(niche["id"], embedding=[0.1, 0.2]))

    assert created["kind"] == "Hypothesis"
    assert created["embedding_status"] == "unavailable"
    assert store.embeddings == []


def test_qd_operators_are_advisory_and_niche_best_requires_active_member():
    store = FakeStore()
    service = HypothesisQDService(store)
    niche = service.niche_create("project", "routing", "Routing", {"family": "routing"})
    hypothesis = service.create("project", draft(niche["id"]))
    niche["status"] = "DEFERRED"

    selected = service.niche_set_best(niche["id"], hypothesis["id"], "Cheapest falsifier first")
    generated = HypothesisGenerator().run({"drafts": [draft(niche["id"])]})
    reflected = ScientificReflector().run({"draft": draft(niche["id"])})
    criticized = ProximityCritic().run(
        {"matches": [{"id": "dead", "flags": ["DESCENDANT_OF_DEAD_IDEA"]}]}
    )

    assert selected["data"]["active_best_hypothesis_id"] == hypothesis["id"]
    assert selected["status"] == "DEFERRED"
    assert generated.advisory is True
    assert reflected.operator == "ScientificReflector"
    assert criticized.accepted is False
