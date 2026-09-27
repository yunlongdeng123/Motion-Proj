"""检查窗口参数化对原r18首窗的真实输入保持一致。"""
import sys
sys.path.insert(0,'/root/autodl-tmp/work')
from hybrid_neighbor_temporal_extension import build,BASE,ROOT
from hybrid_common import dump
import numpy as np
from PIL import Image
e,v=build(65);rows=[]
for i,t in enumerate(e.payloads['neighbor'][0]):
 actual=((t.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype('uint8');expected=np.array(Image.open(BASE/'r18/condition'/f'{i:05}.png'))
 assert np.array_equal(actual,expected),(i,int(np.max(np.abs(actual.astype(int)-expected))))
 assert np.array_equal(e.im[i],np.array(Image.open(BASE/'r15/input'/f'{i:05}.png')))
 assert np.array_equal(e.masks[i],np.array(Image.open(BASE/'r15/condition'/f'{i:05}.png'))>0)
 rows.append(dict(frame=65+i,r18_condition_maxdiff=0,original_and_mask_exact=True))
dump(ROOT/'compatibility_validation.json',dict(checks=rows,source='Actual saved r18 conditions and r15 RGB/masks, no GPU replay',human_verdict=None));print('R22_COMPATIBILITY_PASS',len(rows))
