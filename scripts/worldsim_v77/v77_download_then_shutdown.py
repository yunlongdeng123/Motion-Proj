"""用户本次授权：恢复DriveEditor权重后关机。单次顺序作业，无定时器。"""
import datetime,fcntl,json,os,pathlib,signal,subprocess,sys,time

REPO=pathlib.Path('/root/autodl-tmp/motion_proj_v77')
RUN=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DOWNLOAD-POWEROFF-20260927/r1')
DOWNLOAD=pathlib.Path('/root/autodl-tmp/work/v77_driveeditor_restore')
TARGET=pathlib.Path('/root/autodl-tmp/models/worldsim_v75_downstream_bench/driveeditor/downloads/model.safetensors')
TOTAL=12059467678
RUN.mkdir(parents=True,exist_ok=True)
lock=(RUN/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
STATE=RUN/'state.json'
def write(state,**extra):
    row={'task_id':'WS-V77-DOWNLOAD-POWEROFF-20260927','run_id':'r1','pid':os.getpid(),'state':state,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'用户本轮要求：下载好后把远端AutoDL关机','inference_enabled':False,**extra}
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');temp.replace(STATE)
    print(json.dumps(row,ensure_ascii=False),flush=True)

def processes():
    rows=[]
    for d in pathlib.Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            raw=(d/'cmdline').read_bytes();parts=[x.decode(errors='replace') for x in raw.split(b'\0') if x]
            stat=(d/'stat').read_text();tail=stat[stat.rfind(')')+2:].split()
            rows.append({'pid':int(d.name),'ppid':int(tail[1]),'state':tail[0],'comm':(d/'comm').read_text().strip(),'parts':parts})
        except (OSError,ValueError):pass
    return rows

def busy_processes():
    rows=processes();parents={r['pid']:r['ppid'] for r in rows}
    ignored={os.getpid()};p=os.getppid()
    while p>0 and p not in ignored:ignored.add(p);p=parents.get(p,0)
    infrastructure={'sshd','supervisord','autopanel','proxy','tensorboard','jupyter-lab'}
    busy=[]
    for r in rows:
        parts=r['parts'];comm=r['comm'];pid=r['pid']
        if pid in ignored or pid==1 or r['state']=='Z' or not parts:continue
        if comm in infrastructure:continue
        if comm=='server' and 'tensorboard_data_server' in parts[0]:continue
        if comm in {'bash','sh'} and len(parts)==1:continue
        # 其他计算、下载、渲染、控制器或临时命令一律保守留待检查。
        busy.append({'pid':pid,'comm':comm,'executable':parts[0]})
    return busy

if '--preflight' in sys.argv:
    assert os.environ.get('V77_DOWNLOAD_PROXY')
    assert pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DRIVEEDITOR-COMPARE-20260927/r1/hold_inference_for_shutdown.json').is_file()
    assert 'xargs kill' in pathlib.Path('/usr/bin/shutdown').read_text()
    blockers=busy_processes();assert not blockers,blockers
    write('preflight_passed',download_complete=TARGET.exists(),shutdown_executed=False)
    raise SystemExit(0)

try:
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    assert env.get('V77_DOWNLOAD_PROXY'), '需要本次检查过的代理地址'
    if not TARGET.exists():
        write('downloading',resume=True,total_bytes=TOTAL)
        with (RUN/'download.log').open('a') as log:
            proc=subprocess.Popen(['/root/autodl-tmp/envs/driveeditor/bin/python','/root/autodl-tmp/work/v77_restore_driveeditor.py'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=proc.wait(timeout=7500)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=15);code=124
        if code:raise RuntimeError(f'下载未完成，保留分片，不提前关机；exit={code}')
    assert TARGET.stat().st_size==TOTAL
    from safetensors import safe_open
    with safe_open(TARGET,framework='pt',device='cpu') as f:
        tensor_count=len(list(f.keys()));assert tensor_count>1000
    with TARGET.open('rb') as f:os.fsync(f.fileno())
    os.sync()
    write('download_verified',total_bytes=TOTAL,tensor_count=tensor_count)
    # 在关机前保存并推送当前状态，避免下次误接回旧的自动推理链。
    git_status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    assert not git_status, '存在其他未提交改动，保持开机供人工处理'
    path=REPO/'docs/RESEARCH_STATUS.md'
    text=path.read_text()
    pending='权重状态：正在断点续传；校验完成后关机；本次不启动推理。'
    assert pending in text, '状态已被其他任务更改，停止自动关机'
    backup=RUN/'RESEARCH_STATUS.before_completion.md';backup.write_text(text)
    text=text.replace(pending,f'权重状态：已完整下载并通过safetensors结构检查（{TOTAL:,}字节，{tensor_count}个张量）；已同步落盘，准备执行关机，实际关机状态见run的state.json；本次未启动推理。',1)
    path.write_text(text)
    subprocess.run(['git','add','docs/RESEARCH_STATUS.md'],cwd=REPO,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
    diff=subprocess.check_output(['git','diff','--cached'],cwd=REPO,text=True)
    (RUN/'completion_staged.diff').write_text(diff)
    assert diff.count('diff --git ')==1 and 'docs/RESEARCH_STATUS.md' in diff
    subprocess.run(['git','commit','-m','chore(v77): record verified DriveEditor weights before poweroff'],cwd=REPO,check=True)
    pushenv=dict(env,HTTPS_PROXY=env['V77_DOWNLOAD_PROXY'],HTTP_PROXY=env['V77_DOWNLOAD_PROXY'])
    subprocess.run(['git','push','origin','v77'],cwd=REPO,env=pushenv,check=True,timeout=120)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO)==subprocess.check_output(['git','rev-parse','origin/v77'],cwd=REPO)
    blockers=busy_processes()
    if blockers:
        write('shutdown_blocked_by_other_processes',blockers=blockers)
        raise SystemExit(3)
    supervisors=[r for r in processes() if r['comm']=='supervisord' and '/init/supervisor/supervisor.ini' in r['parts']]
    assert len(supervisors)==1
    # AutoDL的shutdown包装器通过终止supervisord退出实例；不执行其中无关的清空回收站。
    wrapper=pathlib.Path('/usr/bin/shutdown').read_text()
    assert 'supervisord' in wrapper and 'xargs kill' in wrapper
    write('shutdown_requested',checkpoint=str(TARGET),total_bytes=TOTAL,tensor_count=tensor_count,supervisor_pid=supervisors[0]['pid'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    os.sync()
    with open('/proc/1/fd/1','w') as log:
        log.write(f'[{datetime.datetime.now().isoformat()}] user execute shutdown command in container! DriveEditor download verified.\n');log.flush()
    os.kill(supervisors[0]['pid'],signal.SIGTERM)
except Exception as exc:
    write('failed_without_shutdown',error=str(exc))
    raise
