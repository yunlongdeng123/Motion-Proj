"""同一campaign两批×两槽的CPU准入；不改变训练/损失/随机数。"""

import fcntl
import json
import os
from pathlib import Path

PAIR_VERSION = 'V3_PAIRS_2_PLUS_2_R1_20261021'
PAIR_POLICY = dict(version=PAIR_VERSION, batches=[['full', 'real_only'], ['prior_only', 'ownership_only']],
    max_concurrent=2, minimum_launch_free_bytes=8*1024**3, runtime_headroom_warning_bytes=4*1024**3,
    failure_policy='stop_new_admissions_drain_healthy_peers_preserve_no_retry',
    existing_full='adopt_live_without_restart', global_lock='inherited_full_then_coordinator_exclusive')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def process(pid):
    try:
        root = Path('/proc')/str(int(pid))
        fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), ppid=int(fields[1]), start_ticks=int(fields[19]),
            argv=(root/'cmdline').read_bytes().decode().strip('\0').split('\0'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches_worker(info, campaign, arm, mode):
    if not info:
        return False
    argv = info['argv']
    def option(key, expected):
        return key in argv and argv.index(key)+1 < len(argv) and argv[argv.index(key)+1] == str(expected)
    return (str(campaign/'run_v3_arm.py') in argv and option('--plan', campaign/'plan.json')
            and option('--arm', arm) and option('--mode', mode)
            and option('--output', campaign/'arms'/arm/('gate_001' if mode == 'gate' else 'run30k')))


def validate_pair_plan(plan, approval):
    if (plan.get('pair_policy') != PAIR_POLICY or plan.get('max_concurrent_training') != 2
            or approval.get('pair_policy') != PAIR_POLICY or approval.get('max_concurrent_training') != 2):
        raise RuntimeError('2+2 resource policy and human approval must match exactly')


def change_lease(campaign, update):
    with (campaign/'pair_admission.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = campaign/'pair_lease.json'
        value = read(path) if path.exists() else {}
        update(value)
        write(path, value)
        return value


def validate_peer_admission(lease, campaign, arm, mode, pid, parent_pid, lookup=process):
    owner = lookup(parent_pid)
    if (not owner or lease.get('coordinator_pid') != parent_pid
            or lease.get('coordinator_start_ticks') != owner['start_ticks']
            or str(campaign/'run_v3_pairs.py') not in owner['argv']):
        raise RuntimeError('parallel entry requires its live approved coordinator parent')
    slot = lease.get('slots', {}).get(arm, {})
    if slot.get('phase') != mode or slot.get('pid') not in (None, pid):
        raise RuntimeError('parallel arm/phase is not the coordinator admitted slot')
    batch = PAIR_POLICY['batches'][lease['batch_index']]
    if arm not in batch or set(lease['slots'])-set(batch) or len(lease['slots']) > 2:
        raise RuntimeError('parallel entry exceeds registered pair')
    holder = lease.get('global_guard', {})
    holder_info = lookup(holder.get('pid', -1))
    if not holder_info or holder_info['start_ticks'] != holder.get('start_ticks'):
        raise RuntimeError('global GPU guard holder is no longer live')
    if holder['pid'] != parent_pid and not matches_worker(holder_info, campaign, 'full', 'train'):
        raise RuntimeError('only the adopted full worker may hold inherited global guard')
    allowed = set()
    for name, item in lease['slots'].items():
        peer = lookup(item.get('pid', -1)) if item.get('pid') else None
        if peer:
            if peer['start_ticks'] != item.get('start_ticks') or not matches_worker(peer, campaign, name, item['phase']):
                raise RuntimeError('live peer identity differs from admitted arm')
            allowed.add(peer['pid'])
    return allowed


def acquire_pair_entry(arguments, plan, applications_fn):
    campaign = Path(plan['campaign']).resolve()
    slot_lock = (campaign/'arms'/arguments.arm/'entry.lock').open('a+')
    fcntl.flock(slot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own = process(os.getpid())
    def claim(lease):
        allowed = validate_peer_admission(lease, campaign, arguments.arm, arguments.mode, os.getpid(), os.getppid())
        applications = applications_fn()
        outsiders = set(applications)-allowed-{os.getpid()}
        if outsiders:
            raise RuntimeError('GPU has unapproved compute processes: '+str(sorted(outsiders)))
        lease['slots'][arguments.arm].update(pid=os.getpid(), start_ticks=own['start_ticks'])
    change_lease(campaign, claim)
    return slot_lock
