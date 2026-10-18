"""Only the two explicitly registered children may share the guarded GPU."""
from pathlib import Path
import os,json,fcntl
E=Path(__file__).resolve().parent
def read_guard():
 g=json.loads((E/'parallel_group_guard.json').read_text());assert g['state']=='ACTIVE'
 assert Path('/proc') .joinpath(str(g['guardian_pid'])).exists()
 return g
def shared_priority():
 import priority
 p=priority._unshared_inspect_priority()
 try:g=read_guard()
 except (AssertionError,OSError,ValueError):
  p.update(allowed=False,shared_guard_lost=True);return p
 accepted=[];other=[]
 for row in p['other_compute_processes']:
  pid=row['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
  expected={str(E/'run_A_native_shared.py').encode(),str(E/'run_B_native_shared.py').encode()}
  if pid in g['peer_pids'] and expected.intersection(cmd):accepted.append(row)
  else:other.append(row)
 p.update(allowed=not p['streetcrafter_processes'] and not other and p['query_error'] is None,
          authorized_parallel_peers=accepted,other_compute_processes=other)
 return p
def shared_slot(role):
 g=read_guard();assert os.getpid() in g['peer_pids']
 p=shared_priority();assert p['allowed'],p
 f=(E/('parallel_'+role+'.lock')).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def install_priority():
 import priority
 priority._unshared_inspect_priority=priority.inspect_priority
 priority.inspect_priority=shared_priority
 priority.require_gpu_slot=lambda path:(shared_slot('A'),shared_priority())
