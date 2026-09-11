"""网络输入与训练监督分开加载，按真实帧任务索引取数据。"""
import json
from pathlib import Path
import numpy as np
def load_input(episode):
    with np.load(episode['input'],allow_pickle=False) as data:result={k:data[k] for k in data.files}
    result['metadata']=json.loads((Path(episode['input']).parent/'input_metadata.json').read_text());return result
def load_fit_supervision(episode):
    if episode['phase'] not in ['FIT_TRAIN','FIT_VAL']:raise ValueError('训练监督只能读取 FIT 任务')
    with np.load(episode['supervision'],allow_pickle=False) as data:return {k:data[k] for k in data.files}
