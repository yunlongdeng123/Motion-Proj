from motion_proj.resim.scenario_effect import build_counterfactual_pair


def test_pair_requires_same_actor_and_opposite_effects():
    positive = {"positive": True, "label_transition": "0->1", "scenario_effect_hash": "p"}
    negative = {"negative": True, "label_transition": "0->0", "scenario_effect_hash": "n"}
    pair = build_counterfactual_pair(
        scene_id="003", source_actor_id=35,
        positive_proposal_id="p0", negative_proposal_id="n0",
        positive_effect=positive, negative_effect=negative,
    )
    assert pair["source_actor_id"] == 35
    assert pair["realized_effect"] == {"positive": "0->1", "negative": "0->0"}
    assert len(pair["pair_hash"]) == 64
