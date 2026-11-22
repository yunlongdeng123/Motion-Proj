"""Join immutable official point CP/enum, GT OBBs and original-photo semantics.

CP is read from the original record, not recomputed by the Brown transform.
No source/pose/geometry changes, fitting, frame rejection, or GPU access.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from PIL import Image,ImageDraw


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def stat(x):
    x=np.asarray(x);return dict(count=len(x),median=float(np.median(x)),p95=float(np.quantile(x,.95)),max=float(x.max())) if len(x) else dict(count=0)
def cropbox(uv,size):
    x,y=np.rint(uv).astype(int);w,h=size;left=int(np.clip(x-64,0,w-128));top=int(np.clip(y-64,0,h-128));return [left,top,left+128,top+128]
def boxes(world,observation,tracks):
    hits=[];special=[];near=[]
    for box in observation['boxes']:
        T=np.asarray(box['world_from_actor']).reshape(4,4);half=np.asarray(box['dimensions'])/2;q=(np.asarray(world)-T[:3,3])@T[:3,:3]
        inside=bool((abs(q)<=half).all());footprint=bool((abs(q[:2])<=half[:2]).all());role='deleted_a9' if box['track']==tracks['a9'] else 'retained_a20' if box['track']==tracks['a20'] else 'other_retained_GT'
        item=dict(track=box['track'],role=role,inside_GT_OBB=inside,inside_XY_footprint=footprint,above_GT_box_bottom_m=float(q[2]+half[2]),local_point_m=q.tolist(),distance_to_GT_OBB_m=float(np.linalg.norm(np.maximum(abs(q)-half,0))))
        if inside:hits.append(item)
        if role in ['deleted_a9','retained_a20']:special.append(item)
        near.append(item)
    return dict(containing_GT_boxes=hits,A9_A20_box_readouts=special,nearest_GT_box=min(near,key=lambda x:x['distance_to_GT_OBB_m']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;assert not a.output.exists();start=time.time()
    src=r/'audit/height_pair_Brown_cpu_v1';chain=r/'audit/anomalous_raw_point_chain_v1';B=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
    registry=read(B/'track_registry.json');tracks={v['alias']:k for k,v in registry['actors'].items() if v.get('alias') in ['a9','a20']}
    frames=[60,70,110,145,165];data={};photos={};photo_sem={};observations={};pins=[]
    for raw in frames:
        path=chain/f'{raw:03d}/official_TOP_camera_projection_and_world.npz'
        with np.load(path) as z:data[raw]={k:z[k] for k in z.files}
        ids=data[raw]['lidar_row_id'];assert len(np.unique(ids))==len(ids);data[raw]['lookup']={int(k):i for i,k in enumerate(ids)}
        op=B/f'observations/{raw:03d}.json';observations[raw]=read(op);ip=B/observations[raw]['image_path'];photos[raw]=Image.open(ip).convert('RGB')
        sp=B.parent/f'semantics/{raw:03d}.png';photo_sem[raw]=np.asarray(Image.open(sp));assert photo_sem[raw].shape==photos[raw].size[::-1]
        pins.extend([pin(path),pin(op),pin(ip),pin(sp)])
    a.output.mkdir(parents=True);all_rows=[];per_pair=[];preview=[]
    for pair_folder in sorted(src.glob('*_to_*')):
        points=read(pair_folder/'pairs.json');label=pair_folder.name;out=a.output/label;out.mkdir();rows=[];cp_errors=[]
        for q in points:
            result=dict(pair=label,pair_index=q['pair_index'],XY_distance_m=q['XY_distance_m'],delta_worldXYZ=q['delta_worldXYZ'],mutual_nearest=q['mutual_nearest'])
            for role in ['target','reference']:
                rr=q[role+'_raw'];rowid=q[role+'_lidar_row_id'];d=data[rr];i=d['lookup'][rowid];present=bool(d['official_front_present'][i]);cp=d['official_front_uv'][i].astype(float);uv=np.asarray(q[role+'_Brown_uv']);world=np.asarray(q[role+'_worldXYZ'])
                assert abs(d['exported_worldXYZ'][i]-world).max()<1e-5
                pixel=lambda xy: [int(np.rint(xy[0])),int(np.rint(xy[1]))]
                def sample(xy,valid=True):
                    ix,iy=pixel(xy);h,w=photo_sem[rr].shape
                    return int(photo_sem[rr][iy,ix]) if valid and 0<=ix<w and 0<=iy<h else None
                box=boxes(world,observations[rr],tracks)
                point=dict(raw=rr,lidar_row_id=rowid,TOP_encoded=int(d['TOP_encoded'][i]),return_index=int(d['return_index'][i]),official_semantic_enum=int(d['official_semantic'][i]),
                    official_CP_Front_present=present,official_CP_Front_uv=cp.tolist() if present else None,official_CP_all6=d['official_camera_projection'][i].tolist(),Brown_uv=uv.tolist(),
                    Brown_minus_official_CP_px=(uv-cp).tolist() if present else None,photo_semantic_at_official_CP=sample(cp,present),photo_semantic_at_Brown=sample(uv),
                    exported_worldXYZ=world.tolist(),independent_float64_worldXYZ=d['independent_float64_pixel_worldXYZ'][i].tolist(),GT=box)
                result[role]=point
                if present:cp_errors.append(float(np.linalg.norm(uv-cp)))
            rows.append(result);all_rows.append(result)
        order=np.argsort([-abs(q['delta_worldXYZ'][2]) for q in rows])[:12]
        sheet=Image.new('RGB',(840,44+176*int(np.ceil(len(order)/3))),'#101b2a');draw=ImageDraw.Draw(sheet)
        draw.text((8,6),label+' native128px crops: red+ original-record CP; cyan circle Brown; all matches remain in JSON',fill='white')
        for pos,k in enumerate(order):
            q=rows[k];xx=(pos%3)*280;yy=44+(pos//3)*176
            draw.text((xx+2,yy),f"pair{q['pair_index']} XY{q['XY_distance_m']*100:.1f}cm dZ{q['delta_worldXYZ'][2]*100:+.1f}cm",fill='white')
            for col,role in enumerate(['target','reference']):
                item=q[role];rr=item['raw'];cp=item['official_CP_Front_uv'];uv=np.asarray(cp if cp is not None else item['Brown_uv'])
                if not(np.isfinite(uv).all() and 0<=uv[0]<photos[rr].width and 0<=uv[1]<photos[rr].height):continue
                bbox=cropbox(uv,photos[rr].size);crop=photos[rr].crop(bbox);path=out/f"pair{q['pair_index']:03d}_{role}_CP_original_crop.png";crop.save(path)
                assert np.array_equal(np.asarray(Image.open(path)),np.asarray(photos[rr])[bbox[1]:bbox[3],bbox[0]:bbox[2]])
                item['original_pixel_crop']=dict(path=str(path),bbox_xyxy=bbox,sha256=sha(path),original_pixels_exact=True)
                marked=crop.copy();dd=ImageDraw.Draw(marked)
                if cp is not None:
                    x,y=np.asarray(cp)-bbox[:2];dd.line((x-7,y,x+7,y),fill='#ff5e59');dd.line((x,y-7,x,y+7),fill='#ff5e59')
                x,y=np.asarray(item['Brown_uv'])-bbox[:2];dd.ellipse((x-4,y-4,x+4,y+4),outline='#43e0ed',width=1)
                sheet.paste(marked,(xx+col*138,yy+19));enum=item['official_semantic_enum'];sem=item['photo_semantic_at_official_CP']
                draw.text((xx+col*138,yy+150),f'raw{rr} enum{enum} photo{sem}',fill='white')
        sheet.save(out/'contactsheet.png');preview.append(dict(pair=label,path=str(out/'contactsheet.png'),sha256=sha(out/'contactsheet.png')))
        (out/'point_certificates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        per_pair.append(dict(pair=label,pairs=len(rows),Brown_vs_original_CP_pixel_distance=stat(cp_errors),
            official_point_enum_hist={str(k):sum(q[s]['official_semantic_enum']==k for q in rows for s in ['target','reference']) for k in sorted({q[s]['official_semantic_enum'] for q in rows for s in ['target','reference']})},
            photoCP_semantic_hist={str(k):sum(q[s]['photo_semantic_at_official_CP']==k for q in rows for s in ['target','reference']) for k in sorted({q[s]['photo_semantic_at_official_CP'] for q in rows for s in ['target','reference'] if q[s]['photo_semantic_at_official_CP'] is not None})},
            points_inside_retained_GT_OBB=sum(any(b['role']!='deleted_a9' for b in q[s]['GT']['containing_GT_boxes']) for q in rows for s in ['target','reference']),
            points_inside_deleted_A9_GT_OBB=sum(any(b['role']=='deleted_a9' for b in q[s]['GT']['containing_GT_boxes']) for q in rows for s in ['target','reference'])))
    summary=dict(schema='R10.original_CP_photo_semantic_GT_point_audit.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,actual_input_pins=pins,prior_Brown_pairs=pin(src/'summary.json'),original_chain_summary=pin(chain/'point_chain_summary.json'),
        CP_is_original_record_not_recomputed=True,world_Brown_and_CP_both_preserved=True,pairs=per_pair,previews=preview,
        photo_semantic_interpretation='Original photo segmentation proxy; road0/car13. Camera projection is geometric correspondence and does not certify lack of image occlusion.',
        GT_box_interpretation='Original same-frame observed GT 3D box; point-to-bottom signed readout is meaningful with footprint flag, not dense road GT.',
        point_enum_interpretation='Original official point enum independently decoded; road18/lane19 do not alone exclude upper/lower road surfaces.',
        cause='UNRESOLVED; no pose correction, frame deletion or heightfield compensation authorized by these tables',seconds=time.time()-start)
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=True,summary=pin(a.output/'summary.json'),pairs=per_pair,previews=preview),ensure_ascii=False))


if __name__=='__main__':main()
