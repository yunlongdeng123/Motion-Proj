"""统一输入清理：仅保留包含同帧实测正提示的 SAM2 连通域。

不膨胀轮廓，不填洞，不以框替代轮廓。候选原件始终保留。
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.candidates/'receipt.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for row in receipt['results']:
        src=a.candidates/row['mask'];arr=np.array(Image.open(src))>0
        components,n=label(arr,structure=np.ones((3,3),int));anchors=[]
        for x,y in row['point_support'].get('positive',[]):
            x=int(round(x));y=int(round(y))
            if 0<=y<arr.shape[0] and 0<=x<arr.shape[1] and components[y,x]>0:anchors.append(int(components[y,x]))
        keep=np.isin(components,anchors) if anchors else np.zeros_like(arr)
        dst=a.output/row['mask'];dst.parent.mkdir(exist_ok=True);Image.fromarray(np.uint8(keep)*255).save(dst)
        rows.append(dict(raw=row['raw'],alias=row['alias'],before=int(arr.sum()),after=int(keep.sum()),components=n,anchored_components=sorted(set(anchors))))
    receipt['component_filter']=dict(rule='8-connected components containing at least one same-frame measured positive prompt; no dilation/fill',rows=rows,adapter_change=True,unfiltered_source=str(a.candidates))
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(dict(status='PASS_ANCHORED_COMPONENT_FILTER',masks=len(rows),removed=sum(x['before']-x['after'] for x in rows))))
if __name__=='__main__':main()
