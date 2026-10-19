"""共享GPU的窄白名单准入与保留全部节点的容量预留。"""
from pathlib import Path
import fcntl,json,os,subprocess,time

def read(path):
    for attempt in range(10):
        try:return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.05)

def process(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    try:
        fields=(p/'stat').read_text().split(') ',1)[1].split()
        return dict(command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip(),
                    starttime_ticks=int(fields[19]),parent_pid=int(fields[1]),state=fields[0])
    except FileNotFoundError:return None

def collective_storage_required(group):
    registry=read(group/'registry.json');cfg=registry['baseline_config']
    pending=0
    required=8e9
    for root in [Path(registry['main_campaign']),*(group/name for name in registry['arm_queue'])]:
        static=cfg['densify_until_num_points'];dynamic=cfg['dynamic_densify_until_num_points']
        status=root/'run30k/status.json'
        if status.exists():
            s=read(status)
            if s.get('iteration',0)>max(cfg['densify_until_iter'],cfg['dynamic_densify_until_iter'],cfg['sgcm_end']):
                static=s['static_points'];dynamic=s['dynamic_points']
        node=(static*640+dynamic*800+230e6)*1.1
        for step in range(5000,30001,5000):
            event=root/'run30k/checkpoint_events'/f'iteration_{step:06d}.json'
            if not event.exists() or read(event).get('status')!='complete':
                pending+=1;required+=node
    return required

def shared_admission(output,manifest):
    group=output.parent.parent
    own=output.parent/'worker.lock';lock=own.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for attempt in range(50):
        registry=read(group/'registry.json')
        if str(os.getpid()) in registry['children']:break
        time.sleep(.1)
    else:raise RuntimeError('worker not registered by fixed group controller')
    parent=process(registry['controller_pid'])
    if not parent or parent['starttime_ticks']!=registry['controller_starttime_ticks'] or os.getppid()!=registry['controller_pid']:
        raise RuntimeError('authorized controller identity differs')
    if 'run_road_ablation_group.py' not in parent['command']:raise RuntimeError('unexpected controller entry')
    registration=registry['children'][str(os.getpid())]
    if process(os.getpid())['command']!=' '.join(registration['command']):raise RuntimeError('worker command differs')
    if registration['arm']!=read(manifest)['variant']:raise RuntimeError('wrong arm registration')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    allowed={registry['main_pid'],*map(int,registry['children'])}
    actual={int(x.strip()) for x in apps.splitlines() if x.strip()}
    if not actual.issubset(allowed):raise RuntimeError('unregistered GPU process: '+str(actual-allowed))
    main=process(registry['main_pid'])
    if main and main['state']!='Z' and (main['starttime_ticks']!=registry['main_starttime_ticks'] or 'source_bundle/run_candidate.py' not in main['command']):
        raise RuntimeError('main PID reused or entry differs')
    if main and main['state']=='D':raise RuntimeError('main blocked in I/O; do not add a GPU task')
    if (Path('/sys/block/md0/md/array_state').read_text().strip() not in ['active','clean'] or
        Path('/sys/block/md0/md/degraded').read_text().strip()!='0' or
        Path('/sys/block/md0/md/sync_action').read_text().strip()!='idle'):
        raise RuntimeError('storage unhealthy; no new GPU task admitted')
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*1024**2
    if free<11*1024**3:raise RuntimeError('shared GPU has less than11GiB free at admission')
    return lock
