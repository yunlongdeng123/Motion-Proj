from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.run_worldsim_v51_f0i_scene1087_15_view_empty_cache_recovery import _validate_config
CONFIG=ROOT/"configs/worldsim_v51/stage_f_f0i_scene1087_15_view_empty_cache_recovery_v1.yaml"
def test_f0i_config_locks_scene1087_15_view_recovery():
    c,rows=_validate_config(CONFIG)
    assert len(rows)==15
    assert [(r["frame"],r["camera"]) for r in rows]==[(f,cam) for f in [0,40,80,120,160] for cam in [0,1,2]]
    assert c["execution"]["pre_matmul_empty_cache"] is True
    assert c["execution"]["attempt"]["sam_num_points_per_batch"]==64
    assert c["decision"]["full_materialization_authorized"] is False
