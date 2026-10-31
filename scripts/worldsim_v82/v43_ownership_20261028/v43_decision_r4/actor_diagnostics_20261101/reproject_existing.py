"""保留旧投影证据，仅重建投影边/框片；实测 LiDAR 不重新计算。"""
import json,pathlib,shutil,time
import build_media as m
from PIL import Image,ImageDraw
import numpy as np

def run(scene_key):
    cfg=json.loads((m.BASE/'scenes.json').read_text(encoding='utf8'));s=next(x for x in cfg['scenes'] if x['scene_key']==scene_key)
    inv=next(x for x in json.loads((m.BASE/'raw_inventory.json').read_text()) if x['scene_key']==scene_key)
    root=m.BASE/'generated'/'scenes'/s['context_id'];backup=root/'projection_diagnostics_r1';backup.mkdir(exist_ok=True)
    for name in ('media.json','media.js','front_boxes.mp4','manifest.json'):
        if not (backup/name).exists():shutil.copy2(root/name,backup/name)
    media=json.loads((root/'media.json').read_text(encoding='utf8'));byraw={x['raw']:x for x in media['frames']};receipts=[];start=time.perf_counter()
    raw=m.BASE/'raw'/pathlib.Path(inv['path']).name;enc=m.encoder(root/'front_boxes_r2.mp4')
    for index,frame,receipt in m.records(raw):
        row=byraw[index];front=next(x for x in frame.images if x.name==1);model=m.camera_model(frame,front)
        ego=np.array(frame.pose.transform).reshape(4,4);visible=m.front_visibility(frame,model)
        labels={x.id:x for x in frame.laser_labels if x.type==1}
        im=Image.open(root/'frames'/f'{index:06d}.jpg').convert('RGB');draw=ImageDraw.Draw(im)
        for box in row['boxes']:
            track=box['track'];_,_,_,corners,edges,_,_=m.box_geometry(labels[track],ego,model);official=visible.get(track)
            if official is None:edges=[]
            box.update(corners2D=corners,edges=edges,official_FRONT_visible=official is not None,
                       projection_comparison=m.projection_comparison(edges,official) if official is not None else None)
            assert official is not None or not edges
            if edges:
                color=m.color(track)
                for edge in edges:draw.line(edge,fill=color,width=2)
                draw.text((min(e[0] for e in edges)+2,min(e[1] for e in edges)+2),media['actors'][track]['alias'],fill=color,stroke_width=1,stroke_fill='black')
            receipts.append(dict(raw=index,track=track,alias=media['actors'][track]['alias'],official_FRONT_visible=official is not None,
                    edges=len(edges),comparison=box['projection_comparison']))
        draw.text((9,9),f'{scene_key} raw {index:03d} | original RGB + projected 3D boxes',fill='white',stroke_width=1,stroke_fill='black')
        enc.stdin.write(np.asarray(im).tobytes())
    enc.stdin.close();assert enc.wait()==0
    (root/'front_boxes_r2.mp4').replace(root/'front_boxes.mp4')
    media['projection']['rolling_shutter']='provided image pose approximation; no per-row rolling shutter or object-motion compensation'
    media['projection']['drawing_visibility']='raw projected_lidar_labels FRONT track whitelist; finite pinhole frustum before distortion'
    media['projection']['revision']='projection_r2';m.dump(root/'media.json',media)
    media['lidar_contract']['motion_limit']='TOP per-pixel ego-motion compensation only; within-scan actor motion is not compensated'
    m.dump(root/'media.json',media)
    (root/'media.js').write_text('window.ACTOR_MEDIA=window.ACTOR_MEDIA||{};window.ACTOR_MEDIA['+json.dumps(s['context_id'])+']='+json.dumps(media,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
    m.dump(root/'projection_validation.json',receipts)
    vals=[x['comparison']['IoU'] for x in receipts if x['comparison'] and x['comparison']['IoU'] is not None]
    summary=dict(revision='projection_r2',all_non_FRONT_edges_empty=all(x['official_FRONT_visible'] or x['edges']==0 for x in receipts),
                 observations=len(receipts),official_FRONT_observations=sum(x['official_FRONT_visible'] for x in receipts),
                 comparable_AABBs=len(vals),median_AABB_IoU=float(np.median(vals)),p10_AABB_IoU=float(np.quantile(vals,.1)),
                 wall_seconds=time.perf_counter()-start,old_projection_evidence=str(backup))
    m.dump(root/'projection_regression.json',summary);print(json.dumps(summary))
if __name__=='__main__':run('waymo-val-035')
