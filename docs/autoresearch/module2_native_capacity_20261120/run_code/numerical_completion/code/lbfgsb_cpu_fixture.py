"""Synthetic CPU-only cross-process restart fixture; not scene optimization."""
from pathlib import Path
import argparse,json
import numpy as np
from checkpointed_lbfgsb import CheckpointedLBFGSB


def fixture():
    rng=np.random.default_rng(101);A=rng.normal(size=(40,16));target=rng.normal(size=40)
    def evaluate(x):
        d=A@x.reshape(-1)-target
        return float(.5*np.dot(d,d)),(A.T@d).reshape(x.shape),dict(role='synthetic_CPU_fixture')
    return np.full(16,.5),evaluate


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('restored');p.add_argument('next');a=p.parse_args()
    x,evaluate=fixture();solver=CheckpointedLBFGSB(x,evaluate,identity={'fixture':'same'},maxFG=2000)
    solver.load(a.checkpoint);solver.save(a.restored);row=solver.next_accepted();solver.save(a.next)
    print(json.dumps(row))
