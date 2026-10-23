"""显式执行才运行真实 CUDA 生命周期；本轮仅 dry-run，绝不串接正式长跑。"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def commands(campaign, python):
    runner, evaluator = campaign/'tools/run_v41_arm.py', campaign/'tools/evaluate_v41.py'
    common = [python, str(runner), '--campaign', str(campaign), '--arm', 'R', '--mode', 'smoke']
    a, b = campaign/'smoke/continuous_R', campaign/'smoke/split_R'
    return [
        common+['--label', 'continuous_R', '--stop-after', '1000', '--execute'],
        common+['--label', 'split_R', '--stop-after', '500', '--execute'],
        common+['--label', 'split_R', '--stop-after', '1000', '--resume', str(b/'chkpnt500.pth'), '--execute'],
        [python, str(campaign/'tools/compare_v41_states.py'), '--reference', str(a), '--candidate', str(b),
            '--iteration', '1000', '--output', str(campaign/'smoke/exact_resume_comparison.json')],
        [python, str(evaluator), '--campaign', str(campaign), '--arm', 'R', '--run', str(b),
            '--iteration', '1000', '--output', str(campaign/'smoke/evaluation49')]]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--execute', action='store_true'); args = p.parse_args()
    campaign = args.campaign.resolve(); jobs = commands(campaign, sys.executable)
    print(json.dumps(dict(execute=args.execute, commands=jobs, formal_training_started=False,
        smoke_V1_updates=2000, smoke_road_updates=2000, short_nodes=[200, 500, 1000],
        min_actual_wall_seconds=600, note='不足10分钟或任一步失败不签发准入；无自动重放/长跑'), indent=2), flush=True)
    if not args.execute: return
    validation = json.loads((campaign/'cpu_validation.json').read_text())
    if validation['status'] != 'PASS_CPU_PRE_GPU': raise ValueError('CPU 预检查未通过')
    sys.path.insert(0, str(campaign/'engine'))
    from utils.v41_objectives import VERSION, POLICY
    from run_v41_arm import configuration
    formal_config = configuration(campaign, 'R', 'train')  # 同时核验下一阶段真实入口配置，零正式更新。
    root = campaign/'smoke'
    root.mkdir(exist_ok=False)  # 已有现场须按其完整状态核对，不能二次覆盖或自动从零重试。
    started = time.perf_counter(); phases = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    (root/'launch.json').write_text(json.dumps(dict(commands=jobs, formal_updates=0,
        planned_engineering_V1_updates=2000, planned_engineering_road_updates=2000), indent=2))
    try:
        for i, cmd in enumerate(jobs):
            before = time.perf_counter()
            with (root/f'phase_{i}.log').open('x') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            phases.append(dict(index=i, returncode=result.returncode, seconds=time.perf_counter()-before))
            if result.returncode: raise RuntimeError(f'第 {i} 阶段失败，保留完整现场')
        comparison = json.loads((root/'exact_resume_comparison.json').read_text())
        evaluation = json.loads((root/'evaluation49/summary.json').read_text())
        if not comparison['exact_equal'] or evaluation['aggregate49']['frame_count'] != 49:
            raise RuntimeError('完整恢复/49帧评价未通过')
        for name in ('continuous_R', 'split_R'):
            run = root/name
            rows = [json.loads(line) for line in (run/'v41_road_steps.jsonl').read_text().splitlines()]
            if rows[0]['iteration'] != 1 or rows[-1]['iteration'] != 1000:
                raise RuntimeError('道路步数日志不完整')
            if (rows[0]['dc_gradient_l1'] <= 0 or rows[0]['opacity_gradient_l1'] <= 0
                    or any(r['road_points'] != 14034 or not math.isfinite(r['loss']) for r in rows)
                    or not all(r['cuda_peak_allocated_bytes'] > 0 for r in rows)):
                raise RuntimeError('真实梯度/finite loss/固定点数/显存记录检查失败')
            for step in (200, 500, 1000):
                node = json.loads((run/f'v41_checkpoint{step}.json').read_text())
                if len(node['files']) != 5 or not all((run/f).is_file() and (run/f).stat().st_size > 0 for f in node['files']):
                    raise RuntimeError('smoke 完整五件保存失败')
        elapsed = time.perf_counter()-started
        if elapsed < 600: raise RuntimeError('真实 smoke 不足10分钟，需另行登记延长；不以 sleep 凑数')
        status, error = 'PASS_REAL_GPU_LIFECYCLE_SMOKE', None
    except Exception as exc:
        status, error = 'FAILED_OR_INCOMPLETE_SMOKE', str(exc)
    receipt = dict(status=status, error=error, wall_seconds=time.perf_counter()-started, phases=phases,
        version=VERSION, policy=POLICY, formal_config=formal_config,
        smoke_V1_updates=2000 if status.startswith('PASS') else None,
        smoke_road_updates=2000 if status.startswith('PASS') else None,
        actual_update_source='phase_*_returned.json and complete V41 nodes, incomplete phases retained',
        formal_updates=0, formal_training_started=False, quality_accepted=False, human_verdict=None)
    (root/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))
    if status != 'PASS_REAL_GPU_LIFECYCLE_SMOKE': raise SystemExit(1)


if __name__ == '__main__': main()
