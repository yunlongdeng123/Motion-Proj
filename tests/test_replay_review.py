import json

import torch

from motion_proj.replay.review import aggregate_reviews, build_panel


def test_build_panel_keeps_four_labeled_columns():
    base = torch.zeros(2, 3, 4, 6)
    projected = torch.ones_like(base)
    object_mask = torch.ones(2, 1, 1, 2)

    panel = build_panel(base, projected, object_mask)

    assert panel.shape == (2, 4, 24, 3)
    assert panel.dtype.name == "uint8"


def test_aggregate_reviews_promotes_only_after_all_cases(tmp_path):
    (tmp_path / "cases").mkdir()
    for index in range(2):
        (tmp_path / "cases" / f"case-{index}.json").write_text(
            json.dumps({"case_id": f"case-{index}", "case_index": index}), encoding="utf-8"
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "awaiting_reviews"}), encoding="utf-8")
    (tmp_path / "reviews.jsonl").write_text(
        "\n".join(json.dumps({"case_id": f"case-{index}", "object_correction_reasonable": "yes"}) for index in range(2)) + "\n",
        encoding="utf-8",
    )

    summary = aggregate_reviews(tmp_path)

    assert summary["decision"] == "promote"
    assert summary["reasonable_rate"] == 1.0
    assert (tmp_path / "COMPLETE").is_file()
