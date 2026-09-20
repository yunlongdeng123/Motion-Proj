"""验证冻结的对象移除编辑真正进入官方条件raster，且不改初帧。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageChops, ImageDraw
import torch

from closed_loop_bridge import upload_scene
from render_argoverse import render


ROOT=Path('/root/autodl-tmp/runs/worldsim_v75')
OUT=ROOT/'WS-V75-ACTOR-REMOVAL-QUALIFY-01/20260921-r2'
FRAMES=[0,5,30,90]
ARMS=['reference','dvgt_metric','class_prior']


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();out=a.output
    prior=json.loads((out/'result.json').read_text());assert prior['status']=='passed_pending_raster' and prior['raster_probe_required']
    assert not (out/'raster_result.json').exists()
    selected=prior['selected'];base=Path(selected['base']);tr=np.load(base/'trajectory.npz')
    began=time.monotonic();rows=[];rendered={}
    result={'status':'started','frames':FRAMES,'arms':ARMS,'world_model_generation_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    try:
        for arm in ARMS:
            info=selected['state_inputs'][arm]
            for variant,path in [('unedited',Path(info['source'])),('removed',Path(info['edited']))]:
                scene=json.loads(path.read_text());ctx,sid,fit=upload_scene(scene,tr['timestamps_us'],tr['K'])
                images=render(ctx,sid,tr['timestamps_us'][FRAMES],tr['camera_world'][FRAMES])
                rendered[arm,variant]=images
                for frame,image in zip(FRAMES,images):Image.fromarray(image).save(out/f'raster-{arm}-{variant}-{frame:03d}.png')
                del ctx;torch.cuda.empty_cache()
            before,after=rendered[arm,'unedited'],rendered[arm,'removed']
            for i,frame in enumerate(FRAMES):
                mask=np.any(before[i]!=after[i],axis=-1);ys,xs=np.where(mask)
                bbox=None if len(xs)==0 else [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                rows.append({'arm':arm,'frame':frame,'changed_pixels':int(mask.sum()),'difference_bounds_xyxy':bbox,
                             'exactly_equal':bool(np.array_equal(before[i],after[i]))})
        gate={arm:(next(x for x in rows if x['arm']==arm and x['frame']==0)['exactly_equal'] and
                   all(next(x for x in rows if x['arm']==arm and x['frame']==f)['changed_pixels']>=50 for f in FRAMES[1:])) for arm in ARMS}
        sheet=Image.new('RGB',(4*320,3*360),'#142130');draw=ImageDraw.Draw(sheet)
        for row,arm in enumerate(ARMS):
            for col,frame in enumerate(FRAMES):
                i=FRAMES.index(frame);before=Image.fromarray(rendered[arm,'unedited'][i]);after=Image.fromarray(rendered[arm,'removed'][i])
                diff=ImageChops.difference(before,after).point(lambda x:min(255,x*4))
                panel=Image.new('RGB',(320,330),'black');panel.paste(before.resize((320,176)),(0,0));panel.paste(after.resize((160,88)),(0,176));panel.paste(diff.resize((160,88)),(160,176))
                changed=next(x['changed_pixels'] for x in rows if x['arm']==arm and x['frame']==frame)
                ImageDraw.Draw(panel).text((5,272),f'{arm} | frame {frame} | changed={changed}',fill='white')
                ImageDraw.Draw(panel).text((5,294),'top: unedited | lower: removed / x4 diff',fill='#a9c4d2')
                sheet.paste(panel,(col*320,row*360+25))
            draw.text((5,row*360+5),arm,fill='white')
        sheet.save(out/'raster-probe-review.jpg',quality=94)
        passed=all(gate.values())
        result.update(status='passed' if passed else 'failed',rows=rows,arm_gates=gate,
                      initial_rgb_exact_existing_conditioning=True,post_event_input_changed=passed,
                      generation_admitted=passed,condition_render_calls=len(ARMS)*2,
                      rendered_condition_frames=len(ARMS)*2*len(FRAMES))
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc,torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__,error=str(exc),generation_admitted=False)
        raise
    finally:
        result.update(wall_s=time.monotonic()-began,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      world_model_generation_calls=0)
        save(out/'raster_result.json',result)
    prior.update(status='qualified' if result['generation_admitted'] else result['status'],generation_admitted=result['generation_admitted'],
                 raster_result=str(out/'raster_result.json'))
    save(out/'result.json',prior)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
