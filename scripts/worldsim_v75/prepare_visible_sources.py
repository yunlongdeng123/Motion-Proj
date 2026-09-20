"""先冻结新来源与可见候选排序，再查真实帧；不读取模型结果。"""
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from prepare_argoverse import ROOT,CAMERA,crop_image

P=Path('/root/autodl-tmp/motion_proj')
OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-VISIBLE-DEV-01/20260920-r1')
PY='/root/autodl-tmp/envs/worldsim-v75/bin/python'

def main():
    assert not OUT.exists()
    complete=[d for d in sorted(ROOT.iterdir()) if d.is_dir() and (d/'annotations.feather').exists()
              and len(list((d/'sensors/cameras'/CAMERA).glob('*.jpg')))>=200]
    audit=[];logs=[]
    for directory in complete:
        check=subprocess.run(['rg','-l','-F',directory.name,str(P/'docs')],capture_output=True,text=True)
        assert check.returncode in [0,1]
        refs=[str(Path(x).relative_to(P)) for x in check.stdout.splitlines()]
        audit.append({'log_id':directory.name,'prior_document_references':refs})
        if not refs:logs.append(directory.name)
        if len(logs)==4:break
    assert len(logs)==4,'不扩大到已曝光日志'
    OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-VISIBLE-DEV-01','run_id':'20260920-r1','status':'frozen',
              'frozen_utc':datetime.now(timezone.utc).isoformat(),'logs':logs,'exposure_audit':audit,
              'selection':'first four lexicographic complete local logs absent from project docs',
              'boundary':'new development window; undocumented prior exposure and training overlap unknown; not a final independent test',
              'geometry':'unchanged source screen: same 0.5s start, full projected visibility at five fixed times, >=48x32px, depth5..60m, >=6 causal target-core LiDAR points',
              'candidate_order':'at most six geometry-eligible vehicles per log, descending initial projected box area, UUID tie-break; all candidates recorded before real-image review',
              'target_selection':'within frozen candidate order, first with clean visible contour at all five real times AND unchanged detector/RAFT measurement admission; lock before any reconstruction output',
              'measurement':'detector IoU>=0.3 and score>=0.25 on all five real frames; real initial +/-4px translation checks; RAFT FB<=2px and >=8 points and >=25% original support at all five times',
              'model_budget':'first two logs with a selected target in frozen log order; other logs remain unused for model execution; no replacement after seeing reconstruction',
              'readout':'same DVGT ego-range + calibrated rays, extra fixed GT size/yaw; ordinary bbox and extra global/target LiDAR separately reported',
              'generation':'for locked model cases with reliable raw readout/reference, seed42 GT / DVGT / ordinary bbox / extra target LiDAR; include good cases; no model-error ranking, extra seeds, or error amplitude changes',
              'measurement_frames':[0,15,30,45,60],'seed':42,
              'stop':'exactly four new logs and up to six candidates per log; no window/start/threshold changes or expanding source pool; any OOM stops entire queue',
              'human_verdict':None,'failure_ledger_delta':'none'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    subprocess.run([PY,str(P/'scripts/worldsim_v75/screen_natural_sources.py'),'--frozen-protocol',str(OUT/'protocol.json'),'--output',str(OUT/'screen')],check=True,cwd=P)
    screen=json.loads((OUT/'screen/screen_result.json').read_text());all_rows=[]
    for row in screen['logs']:
        log=row['log_id'];raw=ROOT/log;path=OUT/'real'/log;path.mkdir(parents=True)
        candidates=[r for r in row['geometric_candidates'] if r['eligible']]
        candidates.sort(key=lambda r:(-(r['initial_box'][2]-r['initial_box'][0])*(r['initial_box'][3]-r['initial_box'][1]),r['track']))
        selected=candidates[:6]
        files=sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg'));times=np.array([int(f.stem) for f in files])
        intr=pd.read_feather(raw/'calibration/intrinsics.feather').set_index('sensor_name').loc[CAMERA]
        top=(intr.height_px-intr.width_px*704/1280)/2;crop=[0,top,float(intr.width_px),top+intr.width_px*704/1280]
        images=[]
        for f in [0,15,30,45,60]:
            stamp=row['start_ns']+int(round(f/30*1e9));j=np.argmin(abs(times-stamp));assert abs(times[j]-stamp)<=1000
            rgb=crop_image(files[j],crop);rgb.save(path/f'reference-{f:03d}.png');images.append(rgb)
        result={'log_id':log,'candidates':selected,'geometry_eligible_count':len(candidates),'model_output_seen':False,'human_verdict':None}
        (path/'candidates.json').write_text(json.dumps(result,indent=2)+'\n');all_rows.append(result)
        sheet=Image.new('RGB',(1600,max(1,len(selected))*230+28),'#101b2b');draw=ImageDraw.Draw(sheet)
        for j,t in enumerate([0,.5,1,1.5,2]):draw.text((j*320+8,7),f'Real {t:.1f}s',fill='white')
        for i,c in enumerate(selected):
            y=28+i*230;draw.text((5,y+3),f'{log[:8]} | rank {i+1} | target {c["track"][:8]} | core LiDAR {c["target_core_lidar_points"]}',fill='white')
            for j,rgb in enumerate(images):
                b=c['all_boxes'][j];cx,cy=(np.array(b[:2])+b[2:])/2
                # 大目标使用更宽的实际图像裁剪，保持同一行的裁剪尺度。
                width=max(360,max(v[2]-v[0] for v in c['all_boxes'])*1.3);height=width*.6
                window=[round(cx-width/2),round(cy-height/2),round(cx+width/2),round(cy+height/2)]
                im=rgb.copy();ImageDraw.Draw(im).rectangle(b,outline='yellow',width=2)
                sheet.paste(im.crop(window).resize((320,192)),(j*320,y+24))
        sheet.save(OUT/f'candidates-{log[:8]}.jpg',quality=95)
        print(json.dumps({'log_id':log,'ranked_candidates':len(selected),'geometry_eligible':len(candidates)}),flush=True)
    (OUT/'candidate_index.json').write_text(json.dumps({'status':'complete','logs':all_rows,'reconstruction_calls':0,'generation_calls':0},indent=2)+'\n')

if __name__=='__main__':main()
