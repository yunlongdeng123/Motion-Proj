import importlib.util
import json
import sys
import types
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_dr_v2_drivestudio_registry.py"
)
SPEC = importlib.util.spec_from_file_location("build_dr_v2_drivestudio_registry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
OMEGACONF = types.ModuleType("omegaconf")
OMEGACONF.OmegaConf = object
sys.modules.setdefault("omegaconf", OMEGACONF)
SPEC.loader.exec_module(MODULE)


def test_raw_chains_uses_frozen_standard_json_table(tmp_path: Path) -> None:
    rows = [
        {
            "token": "keep",
            "first_annotation_token": "a0",
            "last_annotation_token": "a1",
            "nbr_annotations": 2,
        },
        {
            "token": "skip",
            "first_annotation_token": "b0",
            "last_annotation_token": "b1",
            "nbr_annotations": 3,
        },
    ]
    (tmp_path / "instance.json").write_text(json.dumps(rows), encoding="utf-8")
    assert MODULE.raw_chains(tmp_path, {"keep"}) == {
        "keep": {
            "first_annotation_token": "a0",
            "last_annotation_token": "a1",
            "nbr_annotations": 2,
        }
    }
