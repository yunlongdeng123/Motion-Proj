"""一次性运行已冻结的 SparseDrive 队列，记录终态后按用户授权关机。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

P = Path('/root/autodl-tmp/motion_proj')
R = Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1')
V = '/root/autodl-tmp/envs/sparsedrive-impact/bin/python'
C = R / 'closeout_20260920'
D = P / 'docs/autoresearch/worldsim_simimpact/closeout_20260920'
SCRIPT = P / 'scripts/worldsim_simimpact'
STAGES = [
    ('real_policy', [V, str(SCRIPT/'run_sparse_native_real.py')], 1800),
    ('real_evaluation', [V, str(SCRIPT/'evaluate_sparse_native_real.py')], 900),
    ('paired_policy', [V, str(SCRIPT/'run_sparse_native_real.py'), '--input-dir', str(R/'scene0004_inputs_r1'), '--out-dir', str(R/'scene0004_outputs_r1')], 1800),
    ('paired_evaluation', [V, str(SCRIPT/'evaluate_sparse_native_real.py'), '--reference-dir', str(R/'scene0004_evaluation_r1'), '--outputs-dir', str(R/'scene0004_outputs_r1')], 900),
]

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def run_bounded(cmd, log, timeout):
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0')
    with log.open('w') as out:
        child = subprocess.Popen(cmd, cwd=P, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            return 124

def publish(state):
    evidence = {}
    for name, rel in [
        ('real_outputs', 'real_outputs_r1/manifest.json'),
        ('real_rows', 'real_outputs_r1/forward_rows.json'),
        ('paired_outputs', 'scene0004_outputs_r1/manifest.json'),
        ('paired_rows', 'scene0004_outputs_r1/forward_rows.json'),
        ('real_results', 'evaluation_r1/real_results.json'),
        ('paired_results', 'scene0004_evaluation_r1/paired_results.json'),
    ]:
        data = read(R/rel)
        if data is not None:
            evidence[name] = data
    state['actual_saved_forward_count'] = sum(len(evidence.get(k, [])) for k in ['real_rows','paired_rows'])
    save(C/'state.json', state)
    save(D/'terminal_state.json', state)
    save(D/'evidence.json', evidence)
    for name, _, _ in STAGES:
        log = C/(name+'.log')
        if log.exists():
            (D/(name+'_tail.txt')).write_text(log.read_text(errors='replace')[-18000:])
    note = ('# SparseDrive 固定队列终态（2026-09-20）\n\n'
            f"状态：{state['status']}；实际已保存前向 {state['actual_saved_forward_count']}/40。"
            '环境/模型错误不计作科学失败；未完成任务不补计。\n\n'
            '六相机输入 → 官方 SparseDrive → 保存原生三秒轨迹 → 固定参考评价 → 保存后关机。\n\n'
            '结果为开环回放，非闭环事故或几何因果证明；真实路线提示、训练集重叠、重建监督与共同裁剪控制保持明示。'
            '既有六日志结果与 Ω 局部几何因果结论不变。人工 verdict=null；failure_ledger_delta=none。\n\n'
            '证据：docs/autoresearch/worldsim_simimpact/closeout_20260920/。完整日志与原始结果保留 runs/ 下。\n\n')
    # 运行器只发布本次 run 的证据，不改协作规则、失败目录或全局状态。
    (D/'README.md').write_text(note, encoding='utf-8')
    paths = [str(D.relative_to(P))]
    with (C/'git.log').open('w') as log:
        for cmd in [
            ['git','add','--',*paths],
            ['git','diff','--cached','--check'],
            ['git','commit','-m','research(simimpact): record bounded SparseDrive terminal result'],
        ]:
            subprocess.run(cmd, cwd=P, stdout=log, stderr=subprocess.STDOUT, timeout=60, check=True)
        proc = subprocess.run(['git','-c','http.proxy=','push','origin','research/worldsim-v7.4-h2-generative-surface'], cwd=P, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        state['push_returncode'] = proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execute-and-shutdown', action='store_true')
    args = ap.parse_args()
    if not args.execute_and_shutdown:
        print(json.dumps({'stages':STAGES, 'max_compute_minutes':90, 'shutdown':['bash','/usr/bin/shutdown','-h','now']}, indent=2))
        return
    C.mkdir(parents=True, exist_ok=True)
    lock = (C/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (C/'state.json').exists():
        raise RuntimeError('One-shot state exists; no automatic restart or overwrite.')
    state = {'task_id':'WS-SIM-SPARSE-NATIVE-CLOSEOUT-01','run_id':'20260920-r1','pid':os.getpid(),
             'started_unix':time.time(),'status':'running','stages':[], 'human_verdict':None,
             'authorization':'User requested closeout and shutdown after remote tasks; terminal failures also close out.',
             'expected_forward_count':40,'new_training':False,'automatic_retry':False}
    save(C/'state.json',state)
    try:
        assert read(R/'registration.json')['runtime_prepared']
        for name, cmd, timeout in STAGES:
            row = {'name':name,'command':cmd,'started_unix':time.time(),'timeout_seconds':timeout}
            state['stages'].append(row)
            save(C/'state.json',state)
            row['returncode'] = run_bounded(cmd,C/(name+'.log'),timeout)
            row['ended_unix'] = time.time()
            save(C/'state.json',state)
            if row['returncode']:
                raise RuntimeError(f"{name} failed with exit {row['returncode']}; dependent stages skipped")
        state['status']='completed'
    except Exception:
        state['status']='failed'
        state['error']=traceback.format_exc()
    state['ended_unix']=time.time()
    try:
        publish(state)
    except Exception:
        state['archive_error']=traceback.format_exc()
    state['shutdown_not_before_unix']=time.time()+120
    save(C/'state.json',state)
    # 给本地同步结果留两分钟；不是无限重试或自动研究恢复。
    time.sleep(120)
    try:
        gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
        state['remaining_gpu_pids']=gpu.stdout.strip()
        if gpu.stdout.strip():
            state['shutdown_status']='deferred_unrelated_gpu_job'
        else:
            state['shutdown_status']='command_requested'
            save(C/'state.json',state)
            os.sync()
            proc=subprocess.run(['bash','/usr/bin/shutdown','-h','now'],timeout=30)
            state['shutdown_command_returncode']=proc.returncode
    except Exception:
        state['shutdown_error']=traceback.format_exc()
    save(C/'state.json',state)

if __name__=='__main__':
    main()
