from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_f0g_target_tensor_allocator_instrumentation import _trace_command
from scripts.run_worldsim_v51_f0j_fresh_45_view_empty_cache_materialization import _validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_f_f0j_fresh_45_view_empty_cache_materialization_v1.yaml"


def test_f0j_config_locks_fresh_three_scene_45_view_recovery():
    config, records = _validate_config(CONFIG)
    assert len(records) == 45
    assert [row["input_group"] for row in config["execution"]["attempts"]] == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    assert config["execution"]["pre_matmul_empty_cache"] is True
    assert config["locks"]["reuse_r035_partial"] is False
    assert config["decision"]["quality_read"] is False
    assert config["decision"]["actor_identity_alignment_read"] is False


def test_f0j_all_scene_commands_preserve_batch64_and_enable_empty_cache():
    config, _ = _validate_config(CONFIG)
    for attempt in config["execution"]["attempts"]:
        command = _trace_command(config, attempt, Path("/input"), Path("/output"), Path("/trace.json"))
        assert "--pre-matmul-empty-cache" in command
        assert command[command.index("--SAM_NUM_POINTS_PER_SIDE") + 1] == "32"
        assert command[command.index("--SAM_NUM_POINTS_PER_BATCH") + 1] == "64"
