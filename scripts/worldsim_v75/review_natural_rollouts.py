"""保存真实、GT条件、自然读出与额外观测修复的实际视频和配对指标。"""
import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw
from prepare_argoverse import ROOT,CAMERA,crop_image
from run_natural_rollouts import OUT,VARIANTS

LABELS={'gt_clean':'GT condition','dvgt_metric':'DVGT + known rays','dvgt_lidar_scaled':'+ global LiDAR scale',
        'ordinary_bbox':'RGB box fit + size/yaw','reference_lidar':'Extra target LiDAR'}

def main():
    global OUT,VARIANTS
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=OUT)
    OUT=parser.parse_args().run_dir
    assert not (OUT/'review_result.json').exists(),'拒绝覆盖总结'
    assert json.loads((OUT/'queue_result.json').read_text())['status']=='complete'
    protocol=json.loads((OUT/'protocol.json').read_text());base=Path(protocol['base_dir'])
    VARIANTS=protocol['variants']
    reference=json.loads((OUT/'evaluator_reference.json').read_text())
    evaluations={v:json.loads((OUT/v/'evaluation.json').read_text()) for v in VARIANTS}
    clean=evaluations['gt_clean']['frames'];summaries=[]
    for name,result in evaluations.items():
        generation=json.loads((OUT/name/'generate_result.json').read_text())
        with av.open(str(OUT/name/'clean.mp4')) as container:decoded=sum(1 for _ in container.decode(video=0))
        assert decoded==237
        pairs=[]
        for row,c,r in zip(result['frames'],clean,reference['frames']):
            f=row['frame'];assert f==c['frame']==r['frame']
            delta=None if row['match'] is None or c['match'] is None else (np.array(row['match']['center'])-c['match']['center'])
            pairs.append({'frame':f,'to_real_px':row['distance_to_real_center_px'],
                          'paired_clean_displacement_px':None if delta is None else delta.tolist(),
                          'paired_clean_distance_px':None if delta is None else float(np.linalg.norm(delta))})
        distances=[x['to_real_px'] for x in pairs if x['to_real_px'] is not None]
        paired=[x['paired_clean_distance_px'] for x in pairs if x['paired_clean_distance_px'] is not None]
        summaries.append({'variant':name,'valid_matches':result['valid_matches'],'scheduled':result['all_scheduled_frames'],
                          'mean_distance_to_real_px':float(np.mean(distances)) if distances else None,
                          'mean_paired_clean_distance_px':float(np.mean(paired)) if paired else None,'pairs':pairs,
                          'decoded_frames':decoded,'peak_allocated_gib':generation['peak_allocated_gib'],'generation_wall_s':generation['wall_s']})
    data={v:np.load(OUT/v/'clean.npy',mmap_mode='r') for v in VARIANTS}
    # 所有栏共用以真实检测为中心的固定裁剪，不能跟随预测移动掩盖差异。
    sheet=Image.new('RGB',(300*(1+len(VARIANTS)),5*202+32),'#101b2b');draw=ImageDraw.Draw(sheet)
    cols=['Real RGB',*[LABELS[v] for v in VARIANTS]]
    for c,title in enumerate(cols):draw.text((c*300+7,8),title,fill='white')
    for i,row in enumerate(reference['frames']):
        f=row['frame'];cx,cy=row['match']['center'];crop=(round(cx-180),round(cy-108),round(cx+180),round(cy+108))
        images=[Image.open(OUT/f'reference-{f:03d}.png'),*[Image.fromarray(data[v][f]) for v in VARIANTS]]
        y=32+i*202
        for c,im in enumerate(images):sheet.paste(im.crop(crop).resize((300,180)),(c*300,y+22))
        draw.text((6,y+4),f't={f/30:.1f}s | common ROI from real frame',fill='white')
    sheet.save(OUT/'target-comparison.jpg',quality=95)
    bm=json.loads((base/'input_manifest.json').read_text());times=np.load(base/'trajectory.npz')['timestamps_ns']
    files=sorted((ROOT/protocol['log_id']/'sensors/cameras'/CAMERA).glob('*.jpg'))
    source=np.array([int(x.stem) for x in files],np.int64)
    arms=['gt_clean','dvgt_metric','reference_lidar']
    full=Image.new('RGB',(1920,4*294),'#101b2b')
    with av.open(str(OUT/'comparison.mp4'),'w') as writer:
        stream=writer.add_stream('libx264',rate=30);stream.width=1920;stream.height=294;stream.pix_fmt='yuv420p';stream.options={'crf':'19'}
        for f in range(237):
            j=int(np.argmin(abs(source-times[f])));rgb=crop_image(files[j],bm['crop_xyxy'])
            frame=Image.new('RGB',(1920,294),'#101b2b');draw=ImageDraw.Draw(frame)
            for c,im in enumerate([rgb,*[Image.fromarray(data[v][f]) for v in arms]]):
                title='Real RGB (20Hz nearest)' if c==0 else LABELS[arms[c-1]]
                draw.text((c*480+6,7),title+f' | {f/30:.2f}s',fill='white')
                frame.paste(im.resize((480,264)),(c*480,30))
            if f in [0,15,30,60]:full.paste(frame,(0,[0,15,30,60].index(f)*294))
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(frame),format='rgb24')):writer.mux(packet)
        for packet in stream.encode():writer.mux(packet)
    full.save(OUT/'full-comparison.jpg',quality=94)
    with av.open(str(OUT/'comparison.mp4')) as container:assert sum(1 for _ in container.decode(video=0))==237
    result={'status':'complete','variants':summaries,'seed':protocol['seed'],'source_logs':1,'target_count':1,
            'primary_window_seconds':[0,2],'policy_feedback':False,'human_verdict':None,
            'failure_ledger_delta':'none','full_comparison_decoded':237,
            'boundary':'single-scene per-seed result; 2D independent detector; GT future displacements retained; extra target LiDAR repair is not same-budget'}
    (OUT/'review_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'complete','variants':[{k:v for k,v in r.items() if k!='pairs'} for r in summaries]}),flush=True)

if __name__=='__main__':main()
