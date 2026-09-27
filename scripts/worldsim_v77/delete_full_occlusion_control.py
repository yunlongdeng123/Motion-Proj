"""同背景/资产/相机，仅取消深度测试的诊断；不替换主结果。"""
import argparse,json,numpy as np
from PIL import Image
from delete_full_common import ROOT,dump
from delete_full_query import load_rgb,compose_actor,label,Writer
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);a=p.parse_args();base=ROOT/a.scene;dest=ROOT/'review'/a.scene
summary=json.loads((dest/'summary.json').read_text());path=dest/'unoccluded_factual.mp4';assert not path.exists(),'不覆盖控制视频'
writer=Writer(path,(688,384));rows=[]
for item in summary['selection']:
    f=item['frame'];c=item['camera'];bg=load_rgb(base/'background_world'/f'{f:03}'/f'hybrid_cam{c}.png');layer=base/'actor_layers'/f'f{f:03}_cam{c}.png'
    if layer.exists():
        rgba=np.array(Image.open(layer).convert('RGBA'));az=np.load(layer.with_name(layer.stem+'_depth.npy'))
        rgb,row=compose_actor(bg,rgba,az,np.full_like(az,np.inf));rows.append({'frame':f,'camera':c,**row})
    else:rgb=bg
    writer.write(label(rgb,f'{a.scene} CAM{c} f{f:03} | no depth test: diagnostic only'))
writer.close();dump(dest/'occlusion_control.json',{'control':'same background/GLB/camera; disable background depth test only','not_correct_occlusion_ground_truth':True,'frames':writer.count,'rows':rows,'human_verdict':None})
print('OCCLUSION_CONTROL_COMPLETE',a.scene,writer.count)
