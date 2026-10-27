"""Respect the global GPU lease; only a verified round parent may admit siblings."""
from contextlib import contextmanager
from pathlib import Path
import os,json,time

@contextmanager
def slot(round_dir=None):
    if round_dir is None:
        from gpu_slot import exclusive
        with exclusive():yield
        return
    p=Path(round_dir).resolve();assert str(p).startswith('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/first_gpu_review_')
    lease=json.loads((p/'lease.json').read_text());stat=Path('/proc/%d/stat'%lease['pid']).read_text().split();assert stat[2]!='Z' and stat[21]==str(lease['start_ticks'])
    assert os.environ.get('V43_REVIEW_LEASE')==lease['token'],'Wrong GPU lease'
    for _ in range(100):
        lease=json.loads((p/'lease.json').read_text())
        if os.getpid() in lease['admitted_pids']:break
        time.sleep(.05)
    else:raise RuntimeError('Unadmitted GPU sibling')
    yield
