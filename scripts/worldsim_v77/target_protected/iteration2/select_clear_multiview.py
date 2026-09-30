"""多视角提案v2：先保观测大小/无遮挡，再取不同相机，避免只追求角度跨度。"""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import ijson,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import projection,wrap

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def main(root,meta):
    out=root/'multiview';data=read(out/'full_track_catalog.json');samples=read(meta/'sample.json');by_scene=defaultdict(list)
    for s in samples:by_scene[s['scene_token']].append(s)
    candidates=[];wanted=set()
    for r in data['views']:
        b=r['box_xyxy'];w=b[2]-b[0];h=b[3]-b[1]
        if w<128 or h<72 or b[3]+8>=512:continue
        # 相机曝光与最近真实GT keyframe <=55ms；只是保守遮挡排除，不认证像素mask。
        s=min(by_scene[r['scene_token']],key=lambda s:abs(s['timestamp']-r['timestamp_us']))
        if abs(s['timestamp']-r['timestamp_us'])>55000:continue
        r=r|{'nearest_sample_token':s['token']};candidates.append(r);wanted.add(s['token'])
    anns=defaultdict(list)
    with (meta/'sample_annotation.json').open('rb') as f:
        for a in ijson.items(f,'item',use_float=True):
            if a['sample_token'] in wanted:anns[a['sample_token']].append(a)
    safe=[];rejected=[]
    for r in candidates:
        f={'camera_to_world':r['camera_to_world'],'_w2c':np.linalg.inv(r['camera_to_world']),'intrinsics_1024':r['intrinsics_1024']};target=projection(r['actor'],f);b=np.array(r['box_xyxy']);area=(b[2]-b[0])*(b[3]-b[1]);hits=[]
        for a in anns[r['nearest_sample_token']]:
            if a['instance_token']==r['instance_token']:continue
            p=projection(a,f,clip_near=True)
            if p is None or p['near_depth']>target['far_depth']:continue
            c=p['box'];size=np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2]));overlap=float(np.prod(size))/area
            if overlap>.03:hits.append({'instance_token':a['instance_token'],'box_overlap_fraction':overlap})
        if hits:rejected.append({'filename':r['filename'],'instance_token':r['instance_token'],'potential_foreground_hits':hits})
        else:safe.append(r)
    grouped=defaultdict(lambda:defaultdict(list))
    for r in safe:grouped[r['instance_token']][r['camera']].append(r)
    pairs=[]
    for tok,cameras in grouped.items():
        best=[max(rs,key=lambda r:r['area_px']) for rs in cameras.values()]
        for i,a in enumerate(best):
            for b in best[i+1:]:
                span=abs(wrap(a['view_angles_deg'][0]-b['view_angles_deg'][0]))
                if span>=15:pairs.append((min(a['area_px'],b['area_px']),tok,[a,b],span))
    chosen=[];seen=set()
    for score,tok,pair,span in sorted(pairs,key=lambda r:(-r[0],r[1])):
        if tok in seen:continue
        seen.add(tok);chosen.extend(pair)
        if len(seen)==4:break
    for r in chosen:r['selection_revision']='well_observed_multicamera_v2';r['filename']=str(r['filename'])
    dump(out/'full_track_catalog_geometry_only.json',data)
    data['selected']=chosen;data['selection_v2']={'min_bbox_px':[128,72],'max_foreground_GT_box_overlap':.03,'bottom_guard_plus_clearance':72,
        'rank':'maximize the weaker view area, minimum two actual cameras and >=15deg yaw span; not largest angle alone',
        'candidate_views':len(candidates),'conservative_foreground_rejected':len(rejected),'retained_views':len(safe),'selected_instances':len(seen),'selected_views':len(chosen),
        'note':'包络交叠只作保守排除，不等价实际分割失败。新来源还须看真实RGB和实例mask。'}
    dump(out/'full_track_catalog.json',data);dump(out/'selection_v2_geometry_rejections.json',rejected)
    print(json.dumps(data['selection_v2'],ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--meta',type=Path,required=True);a=p.parse_args();main(a.root,a.meta)
