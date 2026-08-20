from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from motion_proj.worldsim_v521.panels import build_view_panel


def test_matched_panel_contains_frozen_layout(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    adgs = tmp_path / "adgs.png"
    streetgs = tmp_path / "streetgs.png"
    mask = tmp_path / "mask.png"
    output = tmp_path / "panel.png"
    Image.fromarray(np.full((45, 80, 3), 100, dtype=np.uint8)).save(target)
    Image.fromarray(np.full((45, 80, 3), 90, dtype=np.uint8)).save(adgs)
    Image.fromarray(np.full((45, 80, 3), 110, dtype=np.uint8)).save(streetgs)
    region = np.zeros((45, 80), dtype=np.uint8)
    region[10:30, 20:50] = 255
    Image.fromarray(region).save(mask)
    result = build_view_panel(
        target_path=target,
        prediction_paths={"adgs": adgs, "streetgs": streetgs},
        dynamic_mask_path=mask,
        output=output,
    )
    assert output.is_file()
    with Image.open(output) as panel:
        assert panel.size == (2800, 253)
    assert result["layout"] == [
        "GT", "AD-GS", "StreetGS", "AD-GS residual x4", "StreetGS residual x4",
        "dynamic union", "boundary L1 r=3",
    ]
    assert result["geometry_tile_status"].startswith("omitted_undefined")
