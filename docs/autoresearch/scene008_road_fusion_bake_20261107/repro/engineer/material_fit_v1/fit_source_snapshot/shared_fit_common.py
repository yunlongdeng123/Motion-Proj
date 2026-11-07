"""Shared immutable experiment utilities; imports no CUDA runtime."""
from pathlib import Path
import copy
import hashlib
import json
import random
import numpy as np

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
OUT=RUN/'engineer/material_fit_v1'
HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(raw for raw in range(89,138) if raw not in HELD)
SEED=20261107


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4194304),b''):h.update(block)
    return h.hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def tree_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if hasattr(x,'detach'):visit(x.detach().cpu().contiguous().numpy())
        elif isinstance(x,np.ndarray):h.update(str((x.shape,x.dtype.str)).encode());h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for key in sorted(x,key=repr):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def cpu_tree(value):
    if hasattr(value,'detach'):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list):return [cpu_tree(item) for item in value]
    if isinstance(value,tuple):return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def rng_state(torch):return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
def restore_rng(torch,value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch_cpu']);torch.cuda.set_rng_state_all(value['torch_cuda'])


class Sampler:
    def __init__(self,smoke=False):
        self.generator=np.random.default_rng(SEED);self.order=self.generator.permutation(BUILD);self.cursor=0;self.epoch=0
        if smoke:self.order=np.array([90,128]+[int(raw) for raw in self.order if raw not in (90,128)])
    def next(self):
        if self.cursor==40:self.order=self.generator.permutation(BUILD);self.cursor=0;self.epoch+=1
        raw=int(self.order[self.cursor]);self.cursor+=1;return raw
    def state(self):return dict(generator=copy.deepcopy(self.generator.bit_generator.state),order=self.order.copy(),cursor=self.cursor,epoch=self.epoch)
    def restore(self,value):
        self.generator.bit_generator.state=value['generator'];self.order=value['order'].copy();self.cursor=value['cursor'];self.epoch=value['epoch']
    def peek(self,n=2):
        state=self.state();result=[self.next() for _ in range(n)];self.restore(state);return result
