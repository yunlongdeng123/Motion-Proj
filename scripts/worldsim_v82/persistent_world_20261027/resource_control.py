"""代理写入真实周额度快照；远端worker只执行保存/禁启，不猜测账号额度。"""
import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')


def below_threshold(snapshot):
    remaining = snapshot.get('remaining_percent')
    return remaining is not None and float(remaining) < 20


def stop_requested(root=ROOT):
    root = Path(root)
    if (root / 'quota_stop_requested.json').exists():
        return True
    path = root / 'quota_state.json'
    return path.exists() and below_threshold(json.loads(path.read_text()))


def require_start(repair_training=False, backend_change=False, root=ROOT):
    root = Path(root)
    assert not stop_requested(root), '周额度资源停止线已触发：不启动新任务，保存后收尾关机'
    policy_path = root / 'research_policy.json'
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    if repair_training:
        assert policy.get('allow_new_repair_training', False), '阶段二优先：不自动恢复新修复方法训练'
    if backend_change:
        assert policy.get('allow_backend_change', False), '固定③进行评测：不自动切换视频后端'
