"""按用户最新授权记录已启动的GPU阶段；只更新状态，不启动其他工作。"""
from common import *


def main():
    state=read(O/'controller_state.json');assert state['stage']=='running'
    pid=int((O/'controller.pid').read_text().strip());os.kill(pid,0)
    evidence={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r48',
        'authorization':'2026-10-07 用户：不用检查了，直接开始gpu任务',
        'controller_pid':pid,'controller_state':state,'device':'NVIDIA GeForce RTX 3090 / 24576 MiB',
        'CPU_ready':True,'duplicate_job_started':False,'training_budget':128,'human_verdict':None}
    dump(O/'GPU_start.json',evidence)
    e=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r48'
    dump(e/'GPU_start.json',evidence)
    dump(e/'progress.json',{'task_id':evidence['task_id'],'run_id':'r48','parent_run':'r47',
        'stage':'GPU_running','phase':state['phase'],'controller_pid':pid,
        'GPU_jobs':1,'max_steps':128,'human_verdict':None,
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02'})
    path=REPO/'docs/RESEARCH_STATUS.md';text=path.read_text(encoding='utf-8')
    old='唯一run `WS-V77-TARGET-PROTECTED-20260929/r48` 已完成CPU准备，停在等待用户开GPU。训练0步、新采样0窗、无后台GPU控制器或定时任务。当前CPU实例在线、无GPU，数据盘600GB约余92GB，无需清理。'
    new='唯一run `WS-V77-TARGET-PROTECTED-20260929/r48` CPU准备已完成；用户最新授权直接启动GPU，3090已可用，单控制器1414正在运行固定短循环：参考编码/响应 → 64步训练和验证 → 最多128步。尚未收口或判断新收益。没有新增定时任务，数据盘约余92GB，无需清理。'
    assert old in text;text=text.replace(old,new)
    text=text.replace('下一步用户启GPU后，从r47 step320先查','当前从r47 step320先查')
    path.write_text(text,encoding='utf-8')
    path=REPO/'docs/EXPERIMENTS.md';text=path.read_text(encoding='utf-8')
    text=text.replace('r47续64/128步固定短循环待GPU','r47续64/128步固定短循环已启动，尚未收口')
    path.write_text(text,encoding='utf-8')
    path=REPO/'docs/v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md';text=path.read_text(encoding='utf-8')
    text+='\n\n### GPU启动（按最新用户授权）\n\n2026-10-07用户明确“直接开始gpu任务”，3090已检测，单控制器PID1414启动，当前执行既定模块探针，随后64/128步短循环。前文CPU阶段记录保留；当前状态以RESEARCH_STATUS与controller_state为准。启动时尚无新语义收益判定，没有定时任务。\n'
    path.write_text(text,encoding='utf-8')
    print('GPU_START_RECORDED',pid)


if __name__=='__main__':main()
