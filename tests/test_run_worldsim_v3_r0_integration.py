from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts import run_worldsim_v3_r0_integration as r0


def load_protocol() -> dict:
    return OmegaConf.to_container(OmegaConf.load(r0.PROTOCOL), resolve=True)


def test_frozen_protocol_schema_is_valid() -> None:
    protocol = load_protocol()
    r0.validate_schema(protocol)
    assert protocol["formal_integration"]["expected_deliverable_count"] == 12
    assert protocol["selected_production_asset"]["method_chain"].endswith(
        "A4-P3-exact-chunk"
    )


@pytest.mark.parametrize(
    "authorization",
    ["training_authorized", "model_inference_authorized", "gpu_launch_authorized"],
)
def test_schema_rejects_compute_authorization(authorization: str) -> None:
    protocol = load_protocol()
    protocol["authorization"][authorization] = True
    with pytest.raises(RuntimeError, match=authorization):
        r0.validate_schema(protocol)


def test_schema_rejects_conclusion_inflation() -> None:
    protocol = load_protocol()
    protocol["final_conclusions"].append("local_refine_supported")
    with pytest.raises(RuntimeError, match="local refine inflation"):
        r0.validate_schema(protocol)


def test_schema_rejects_frozen_input_count_drift() -> None:
    protocol = load_protocol()
    protocol["protocol_inputs"].pop("f0_audit")
    with pytest.raises(RuntimeError, match="protocol input count"):
        r0.validate_schema(protocol)


def test_all_frozen_inputs_are_exact() -> None:
    audit = r0.audit_all_inputs(load_protocol())
    assert audit["all_exact"] is True
    assert audit["protocol_input_count"] == 5
    assert audit["canonical_evidence_file_count"] == 51
    assert audit["selected_asset_file_count"] == 3
    assert audit["visualization_file_count"] == 4
    assert len(audit["rows"]) == 63
    assert len(audit["terminal_checks"]) == 11


def test_chunk_package_payloads_are_exact() -> None:
    audit = r0.verify_chunk_package(load_protocol())
    assert audit["all_payloads_exact"] is True
    assert audit["payload_file_count"] == 158
    assert audit["total_file_count"] == 159
    assert audit["static_asset_count"] == 133
    assert audit["actor_asset_count"] == 24
    assert audit["total_bytes"] == 444_177_055


def test_all_frozen_decisions_match_canonical_evidence() -> None:
    audit = r0.validate_frozen_decisions(load_protocol())
    assert audit["all_passed"] is True
    assert audit["passed_count"] == audit["total_count"] == 23


def test_derived_deliverables_keep_selection_and_claim_boundaries() -> None:
    protocol = load_protocol()
    actor = r0.build_actor_quality(protocol)
    local_refine = r0.build_a3_support(protocol)
    table = r0.main_table(protocol)
    negatives = r0.build_negative_results(protocol)

    assert actor["selected_research_asset"] == "D2-boundary-residual"
    assert actor["not_a_dominance_claim"] is True
    assert set(actor["roles"]) == {"high-support", "boundary-support"}
    assert local_refine["selected_arm"] == "R0-off"
    assert local_refine["formal_training_authorized"] is False
    assert [row["stage"] for row in table] == ["A0", "A1", "F0", "A2", "A3", "A4"]
    assert len(negatives["results"]) == 7
    assert all(negatives["claim_boundary"].values())


def test_final_report_uses_frozen_conclusion_vocabulary() -> None:
    protocol = load_protocol()
    report = r0.build_report(protocol, r0.main_table(protocol))
    for conclusion in protocol["final_conclusions"]:
        assert f"`{conclusion}`" in report
    assert "D2 dominates D1" in report
    assert "complete world model" not in report.lower()
