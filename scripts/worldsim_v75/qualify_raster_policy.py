"""现有真实RGB基线的一次普通地面修正；冻结后保留改善与剩余失败。"""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from raster_ground import RasterGround
from rgb_idm_policy import RGBIDMPolicy, IDMPolicy, MotionTracks
from following_geometry import Route

SOURCE=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-BRAKING-TASKS-01/20260920-r1/interface-audit')
OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-RASTER-POLICY-01/20260920-r1')


def save(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    assert not OUT.exists(); OUT.mkdir(parents=True)
    proto=json.loads((SOURCE/'protocol.json').read_text()); base=Path(proto['base']); terrain=RasterGround(base)
    frozen={'task_id':'WS-V75-RASTER-POLICY-01','run_id':OUT.name,'frozen_utc':datetime.now(timezone.utc).isoformat(),
            'source':str(SOURCE),'base':str(base),'target':proto['target'],
            'role':'engineering development repair, same saved actual detector boxes; not a new source or world-model result',
            'change':'single global ground plane replaced by known official raster height; tracker/IDM unchanged',
            'terrain':terrain.provenance,'frames':[0,15,30,45,60],
            'old_state_gate':{'correct_leader_fraction':.8,'median_abs_gap_error_m':3.,'median_abs_acceleration_difference_mps2':.5},
            'stop':'one ordinary ground correction; preserve original failed task window; no threshold tuning or repeated detector calls',
            'human_verdict':None,'failure_ledger_refs':['V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',frozen)
    original=json.loads((SOURCE/'result.json').read_text()); trajectory=np.load(base/'trajectory.npz')
    policy=object.__new__(RGBIDMPolicy); policy.route=Route(trajectory['ego_world'][:,:3,3]); policy.tracker=MotionTracks()
    policy.idm=IDMPolicy(target_velocity=10.,min_gap_to_lead_agent=1.,headway_time=1.5,accel_max=1.,decel_max=3.)
    rows=[]; font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    sheet=Image.new('RGB',(1280,5*410),'#142334')
    from evaluate_localization import iou
    for i,old in enumerate(original['rows']):
        f=old['frame']; state=old['policy']['ego_state']; camera=trajectory['camera_world'][f]; K=trajectory['K']
        detections=[]
        for d in old['policy']['detections']:
            b=d['box']; left=terrain.contact(b[0],b[3],camera,K); right=terrain.contact(b[2],b[3],camera,K)
            if left is None or right is None: continue
            detections.append({'box':b,'score':d['score'],'class_id':d['class_id'],
                               'world_center':((left+right)/2)[:2].tolist(),'contact_edge_world':[left.tolist(),right.tolist()]})
        policy.tracker.update(detections,old['policy']['timestamp_us']/1e6,state['speed_mps']*np.array([np.cos(state['yaw_rad']),np.sin(state['yaw_rad'])]))
        lead=policy.choose_lead(detections,state); accel=policy.acceleration(state['speed_mps'],lead)
        ref=old['reference_lead']; correct=bool(lead is None and ref is None or lead and ref and iou(lead['box'],old['target_projection']['bounds'])>=.3)
        bounds=old['target_projection']['bounds']; projected_ground=terrain.contact((bounds[0]+bounds[2])/2,bounds[3],camera,K)
        row={'frame':f,'reference_lead':ref,'target_projection':old['target_projection'],'lead':lead,'detections':detections,
             'acceleration_mps2':accel,'acceleration_difference_mps2':accel-old['oracle_acceleration_mps2'],
             'correct_leader':correct,'gap_error_m':abs(lead['gap_m']-ref['gap_m']) if correct and ref else None,
             'extra_GT_bbox_surface_point':None if projected_ground is None else projected_ground.tolist()}
        rows.append(row)
        image=Image.open(SOURCE/f'real-{f:03d}.png').copy(); draw=ImageDraw.Draw(image)
        draw.rectangle(bounds,outline='yellow',width=3)
        if lead: draw.rectangle(lead['box'],outline='#14ddb0',width=3)
        y=i*410; d=ImageDraw.Draw(sheet)
        d.text((8,y+4),f't={f/30:.1f}s | GT gap={ref["gap_m"] if ref else None} | raster policy gap={lead["gap_m"] if lead else None}',font=font,fill='white')
        d.text((8,y+26),f'acceleration={accel:.3f} m/s2 | yellow=GT target / green=policy lead | same saved detections',font=font,fill='white')
        sheet.paste(image.resize((640,352)),(0,y+56)); sheet.paste(image.crop((320,220,1120,640)).resize((640,336)),(640,y+56))
    errors=[r['gap_error_m'] for r in rows if r['gap_error_m'] is not None]
    gap=float(np.median(errors)) if errors else None; acc=float(np.median([abs(r['acceleration_difference_mps2']) for r in rows])); correct=float(np.mean([r['correct_leader'] for r in rows]))
    result={'status':'complete','rows':rows,'correct_leader_fraction':correct,'median_abs_gap_error_m':gap,
            'median_abs_acceleration_difference_mps2':acc,'state_gate_passed':bool(correct>=.8 and gap is not None and gap<=3 and acc<=.5),
            'saved_detector_calls_reused':5,'new_detector_calls':0,'world_model_calls':0,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'result.json',result); sheet.save(OUT/'real-policy-review.jpg',quality=94)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__': main()
