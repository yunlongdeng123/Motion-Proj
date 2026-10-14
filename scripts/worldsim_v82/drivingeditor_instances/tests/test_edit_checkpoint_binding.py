"""防止用真实300步内容冒充30k编辑结果，或跨场景绑定检查点。"""
import pytest
from scripts.worldsim_v82.drivingeditor_instances.render_edits import verify_checkpoint_binding


def test_edit_rejects_wrong_actual_saved_step_and_scene():
    spec = {'scene_id': 'gmc_two_actor_fixture'}
    checkpoint = {'schema': 'drivingeditor_rigid_scene_v1', 'spec': spec, 'step': 300}
    # 即使上层传入名为checkpoint_030000.pt的路径，也必须检查加载内容。
    with pytest.raises(ValueError, match='真实step'):
        verify_checkpoint_binding(checkpoint, spec, expected_step=30000)
    with pytest.raises(ValueError, match='场景绑定'):
        verify_checkpoint_binding(checkpoint, {'scene_id': 'different_scene'}, expected_step=300)
    assert verify_checkpoint_binding(checkpoint, spec, expected_step=300) == 300
