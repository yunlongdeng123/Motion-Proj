"""只做冻结基线跨场景评价，禁止接续新方法训练或更换视频后端。"""
import json
import os
import subprocess
import traceback
from pathlib import Path

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
CAMPAIGN = ROOT.parent
PYTHON = '/root/autodl-tmp/envs/drivingeditor/bin/python'
VIDEO_PYTHON = '/root/autodl-tmp/envs/motionproj/bin/python'
RUNS = {
    'scene0017': CAMPAIGN / 'hybrid30k_20261012/runs/scene0017_000_048_hybrid_30000_seed0_GN11',
    'scene0029': CAMPAIGN / 'followup_scene_prep_20261013/runs/scene0029_069_117_hybrid_30000_seed0_GN11',
}


def main():
    status = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                  phase='phase2_fixed_baseline_evaluation', completed=[])
    path = ROOT / 'phase2_eval_status.json'
    def save():
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(path)
    def run(key, command):
        quota = json.loads((ROOT / 'quota_state.json').read_text())
        policy = json.loads((ROOT / 'research_policy.json').read_text())
        assert quota['remaining_percent'] >= 20 and policy['phase'] == 'phase2_evaluation_first', '停止新任务并快速保存收尾'
        status['phase'] = key
        save()
        with (ROOT / (key + '.log')).open('a') as log:
            process = subprocess.Popen(command, cwd=str(ROOT / 'code'), stdout=log, stderr=subprocess.STDOUT)
            status['child'] = dict(pid=process.pid, start_ticks=Path('/proc/%d/stat' % process.pid).read_text().split()[21], command=command)
            save()
            assert process.wait() == 0, key
        status.pop('child')
        status['completed'].append(key)
        save()
    save()
    try:
        # 先证明适配器未改变原生通路，再全49帧。全部是只读推理，无训练更新。
        for name, checkpoint in RUNS.items():
            run(name + '_adapter_preflight', [PYTHON, str(ROOT / 'code/evaluate_multiscene.py'), '--run', str(checkpoint), '--output', str(ROOT / 'phase2_multiscene/preflight' / name), '--single-frame'])
            run(name + '_native49', [PYTHON, str(ROOT / 'code/evaluate_multiscene.py'), '--run', str(checkpoint), '--output', str(ROOT / 'phase2_multiscene' / name)])
            run(name + '_delete_video49', [VIDEO_PYTHON, str(ROOT / 'code/inpaint_baseline.py'), '--root', str(ROOT / 'phase2_multiscene' / name), '--case', 'delete_all'])
        # 已完成的10K基线只读验收，不等待被撤下的几何10K，也不换③。
        run('completed_repair10k_native49', [PYTHON, str(ROOT / 'code/evaluate_repair_state.py'), '--checkpoint', str(ROOT / 'repair10k/checkpoint_final.pth'), '--output', str(ROOT / 'phase2_multiscene/scene0008_repair10k')])
        run('completed_repair10k_video49', [VIDEO_PYTHON, str(ROOT / 'code/inpaint_baseline.py'), '--root', str(ROOT / 'phase2_multiscene'), '--case', 'scene0008_repair10k'])
        status.update(phase='frozen_baselines_complete_analysis_pending', evaluation_complete=True,
                      new_training_updates=0, next='case-level failure matrix and source-withholding preparation; no automatic phase3')
        save()
    except Exception:
        status.update(phase='engineering_repair_needed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
