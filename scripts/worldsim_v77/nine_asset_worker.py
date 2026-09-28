"""现有冻结Hunyuan，所有scene相同参数；每模型单独进程释放显存。"""
from nine_common import *
import argparse
import hybrid_retained_actor_asset as worker
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--stage',choices=['shape','paint'],required=True);a=p.parse_args()
worker.ROOT=ROOT/a.scene/'asset'
if a.stage=='shape':worker.shape()
else:worker.paint()
