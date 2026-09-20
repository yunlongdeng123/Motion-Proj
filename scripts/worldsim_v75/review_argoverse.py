"""固定时间点检查真实/生成目标检测，并保存RGB—条件—生成图；不是badcase扫描。"""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw
from evaluate_localization import model,predict,match
from prepare_argoverse import OUT,ROOT,LOG,CAMERA,crop_image

FRAMES=[0,30,60,90,120,150,180,210,234]

def target_figure():
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    result=json.loads((OUT/'evaluation.json').read_text())
    condition=np.load(OUT/'conditions.npy',mmap_mode='r')
    generated=np.load(OUT/'clean.npy',mmap_mode='r')
    images=sorted((ROOT/LOG/'sensors/cameras'/CAMERA).glob('*.jpg'))
    ns=np.asarray([int(p.stem) for p in images],np.int64)
    times=np.load(OUT/'trajectory.npz')['timestamps_ns']
    sheet=Image.new('RGB',(1200,9*264),'#151c29')
    for index,row in enumerate(result['frames']):
        f=row['frame'];b=row['reference_projection_bounds']
        cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        crop=(round(cx-100),round(cy-60),round(cx+100),round(cy+60))
        y=index*264
        ImageDraw.Draw(sheet).text((8,y+3),f't={f/30:.2f}s | real={row["real"] is not None}, generated={row["generated"] is not None}',fill='white')
        rgb=crop_image(images[int(np.abs(ns-times[f]).argmin())],manifest['crop_xyxy'])
        for col,im in enumerate([rgb,Image.fromarray(condition[f]),Image.fromarray(generated[f])]):
            sheet.paste(im.crop(crop).resize((400,240)),(col*400,y+24))
    sheet.save(OUT/'target-review.jpg',quality=94)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figures-only',action='store_true')
    if parser.parse_args().figures_only:
        target_figure();return
    out=OUT
    protocol={'task_id':'WS-V75-AV2-BRIDGE-01','run_id':'20260920-r1',
              'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'target':'94dede14-59da-4f09-b016-95f19596ac08','frames':FRAMES,
              'selection':'only vehicle satisfying 151-frame geometric visibility, chosen before generated output inspection',
              'evaluator':'same official FasterRCNN ResNet50 FPN v2 COCO weights as localization',
              'class_id':3,'score_threshold':0.25,'matching_iou_threshold':0.3,
              'minimum_real_matches':6,'minimum_generated_matches':6,
              'calibration_shifts_px':[-4,4],'calibration_max_residual_px':2,
              'paired_metric':'2D detector center difference, generated versus real RGB at exact coincident timestamp',
              'claim_boundary':'clean baseline and measurement admission; reference boxes associate detections, not metric 3D GT',
              'human_verdict':None}
    p=out/'evaluator_protocol.json';assert not p.exists();p.write_text(json.dumps(protocol,indent=2)+'\n')
    manifest=json.loads((out/'input_manifest.json').read_text())
    assert json.loads((out/'generate_result.json').read_text())['status']=='complete'
    video=np.load(out/'clean.npy',mmap_mode='r');cond=np.load(out/'conditions.npy',mmap_mode='r')
    projection=next(p for p in json.loads((out/'projections.json').read_text()) if p['id']==protocol['target'])['projections']
    images=sorted((ROOT/LOG/'sensors/cameras'/CAMERA).glob('*.jpg'))
    ns=np.asarray([int(p.stem) for p in images],np.int64)
    times=np.load(out/'trajectory.npz')['timestamps_ns']
    m=model();rows=[]
    initial=np.asarray(Image.open(out/'initial_rgb.png'))
    b=projection[0]['bounds'];base=match(predict(m,initial),b);calibration=[]
    for dx in [-4,4]:
        shifted=np.zeros_like(initial)
        if dx>0:shifted[:,dx:]=initial[:,:-dx]
        else:shifted[:,:dx]=initial[:,-dx:]
        shifted_bounds=(np.asarray(b)+[dx,0,dx,0]).tolist()
        found=match(predict(m,shifted),shifted_bounds)
        residual=None if base is None or found is None else float(found['center'][0]-base['center'][0]-dx)
        calibration.append({'dx':dx,'center_residual_px':residual})
    calibration_pass=base is not None and all(c['center_residual_px'] is not None and abs(c['center_residual_px'])<=2 for c in calibration)
    sheet=Image.new('RGB',(1920,5*378),'#151c29');sheet_row=0
    crop_sheet=Image.new('RGB',(1200,9*264),'#151c29')
    for index,f in enumerate(FRAMES):
        nearest=int(np.abs(ns-times[f]).argmin());delta=int(ns[nearest]-times[f]);assert abs(delta)<=1000
        real=crop_image(images[nearest],manifest['crop_xyxy'])
        generated=Image.fromarray(video[f]);box=projection[f]['bounds']
        rd=match(predict(m,np.asarray(real)),box);gd=match(predict(m,video[f]),box)
        distance=None if rd is None or gd is None else float(np.linalg.norm(np.asarray(rd['center'])-gd['center']))
        rows.append({'frame':f,'timestamp_delta_ns':delta,'reference_projection_bounds':box,
                     'real':rd,'generated':gd,'paired_center_distance_px':distance})
        print(json.dumps({'frame':f,'real_match':rd is not None,'generated_match':gd is not None,'center_distance_px':distance}),flush=True)
        if f in [0,60,120,180,234]:
            y=sheet_row*378;sheet_row+=1
            ImageDraw.Draw(sheet).text((8,y+5),f'Real RGB | Adapted condition | Generated clean | t={f/30:.2f}s',fill='white')
            for col,im in enumerate([real,Image.fromarray(cond[f]),generated]):
                sheet.paste(im.resize((640,352)),(col*640,y+26))
        # 同一固定参考框中心裁剪三列，保留缺失，不跟随预测重定位。
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        crop=(round(cx-100),round(cy-60),round(cx+100),round(cy+60))
        y=index*264;ImageDraw.Draw(crop_sheet).text((8,y+3),f't={f/30:.2f}s | real={rd is not None}, generated={gd is not None}',fill='white')
        for col,im in enumerate([real,Image.fromarray(cond[f]),generated]):
            crop_sheet.paste(im.crop(crop).resize((400,240)),(col*400,y+24))
    sheet.save(out/'clean-review.jpg',quality=93);crop_sheet.save(out/'target-review.jpg',quality=94)
    with av.open(str(out/'comparison.mp4'),'w') as writer:
        stream=writer.add_stream('libx264',rate=30);stream.width=1920;stream.height=378;stream.pix_fmt='yuv420p';stream.options={'crf':'20'}
        for f in range(237):
            nearest=int(np.abs(ns-times[f]).argmin())
            frame=Image.new('RGB',(1920,378),'#151c29')
            ImageDraw.Draw(frame).text((8,5),f'Real RGB (20Hz nearest) | Adapted GT condition | Generated clean | t={f/30:.3f}s | ref dt={(ns[nearest]-times[f])/1e6:.2f}ms',fill='white')
            for col,im in enumerate([crop_image(images[nearest],manifest['crop_xyxy']),Image.fromarray(cond[f]),Image.fromarray(video[f])]):
                frame.paste(im.resize((640,352)),(col*640,26))
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(frame),format='rgb24')):writer.mux(packet)
        for packet in stream.encode():writer.mux(packet)
    with av.open(str(out/'clean.mp4')) as container:
        decoded=sum(1 for frame in container.decode(video=0))
    assert decoded==237
    real_matches=sum(r['real'] is not None for r in rows);gen_matches=sum(r['generated'] is not None for r in rows)
    values=[r['paired_center_distance_px'] for r in rows if r['paired_center_distance_px'] is not None]
    result={'status':'complete','calibration':calibration,'calibration_passed':calibration_pass,
            'real_matches':real_matches,'generated_matches':gen_matches,'scheduled_frames':len(rows),
            'measurement_admission':calibration_pass and real_matches>=6 and gen_matches>=6,
            'paired_center_distance_mean_px':float(np.mean(values)) if values else None,
            'paired_center_distance_median_px':float(np.median(values)) if values else None,
            'paired_denominator':len(values),'frames':rows,'decoded_frames':decoded,
            'failure_ledger_delta':'none','human_verdict':None,
            'interpretation':'真实/生成检测框的二维描述性差异；即使较大，也不能归因自然重建误差或升级为策略危害'}
    (out/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
