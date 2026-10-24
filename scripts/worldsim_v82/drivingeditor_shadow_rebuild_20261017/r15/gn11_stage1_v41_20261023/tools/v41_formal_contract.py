"""用户已知 smoke 限制后显式授权两组30K；保留原 smoke 失败状态。"""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_authorization(campaign, config=None):
    campaign = Path(campaign)
    auth = read(campaign/'formal_authorization.json')
    assert auth['status'] == 'USER_AUTHORIZED_FORMAL_30K_WITH_REPORTED_SMOKE_LIMITS'
    assert auth['arms'] == ['R', 'B'] and auth['iterations_each'] == 30000
    assert auth['checkpoint_steps'] == [10000, 20000, 30000]
    assert auth['quality_accepted'] is False and auth['strict_trajectory_passed'] is False
    assert auth['campaign'] == str(campaign.resolve())
    assert auth['R_config'] == read(campaign/'configs/R.json')
    if config is not None:
        assert config == auth['R_config']
    smoke = campaign/'smoke'
    receipt = read(smoke/'receipt.json')
    assert receipt['status'] == 'FAILED_OR_INCOMPLETE_SMOKE' and receipt['wall_seconds'] >= 600
    assert receipt['formal_config'] == auth['R_config']
    assert read(smoke/'split_R/native_restore_audit500.json')['status'] == 'PASS_EXACT_CUDA_RESTORE'
    assert read(smoke/'split_R/road_restore_audit500.json')['status'] == 'PASS_EXACT_ROAD_RESTORE'
    assert read(smoke/'split_R/phase_500_1000_returned.json')['iteration'] == 1000
    for label in ('continuous_R', 'split_R'):
        for step in (200, 500, 1000):
            run = smoke/label
            node = read(run/f'v41_checkpoint{step}.json')
            assert node['V1_updates'] == node['road_updates'] == step
            assert len(node['files']) == 5
            assert all((run/name).is_file() and (run/name).stat().st_size > 0 for name in node['files'])
    assert read(smoke/'continuous_evaluation1000/summary.json')['aggregate49']['frame_count'] == 49
    return auth
