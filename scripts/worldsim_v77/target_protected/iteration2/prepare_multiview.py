"""CPU：同一真实instance跨相机/时刻证据索引；不生成新视角、不把3D框当实例mask。"""
import argparse,bisect,json,sys
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from PIL import Image,ImageDraw
from pyquaternion import Quaternion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import transform,projection,view_angles,wrap

def read(p):return json.loads(Path(p).read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')

def interpolate(c,ctx,t,tok):
    times=c['keyframe_timestamps'];hi=bisect.bisect_right(times,t)
    if hi==len(times) and t==times[-1]:hi-=1
    if hi==0 or hi>=len(times) or times[hi]-times[hi-1]>600000:return None
    lo=hi-1;ends=[]
    for j in [lo,hi]:
        a=next((a for a in ctx['frames'][j]['annotations'] if a['instance_token']==tok),None)
        if a is None:return None
        ends.append(a)
    if any(a['visibility_token']!='4' for a in ends):return None
    a,b=ends;u=(t-times[lo])/(times[hi]-times[lo])
    return a|{'translation':((1-u)*np.array(a['translation'])+u*np.array(b['translation'])).tolist(),
              'rotation':Quaternion.slerp(Quaternion(a['rotation']),Quaternion(b['rotation']),amount=u).elements.tolist()}

def main(root,factory):
    cv2.setNumThreads(2);out=root/'multiview';out.mkdir(exist_ok=True);(out/'crops').mkdir(exist_ok=True)
    cs=read(factory/'source_manifest.json')['clips'];ct={c['source_id']:c for c in read(factory/'source_context.json')['clips']}
    qa={c['source_id']:c for c in read(factory/'mask_review/independent_mask_reviews.json')['clips']}
    rows=[];missing=[];reasons=Counter();seen=set()
    for c in cs:
        sid=c['source_id'];q=qa.get(sid,{})
        if q.get('donor_mask_status') not in ('pass','uncertain'):continue
        tok=c['actors'][0]['instance_token'];ctx=ct[sid]
        for k,cf in enumerate(ctx['frames']):
            for channel,s in cf['sensors'].items():
                if not channel.startswith('CAM_') or (tok,s['filename']) in seen:continue
                a=interpolate(c,ctx,s['timestamp'],tok)
                if a is None:reasons['unobserved_or_no_bracket']+=1;continue
                cal=s['calibrated_sensor'];eg=s['ego_pose'];c2w=transform(eg['translation'],eg['rotation'])@transform(cal['translation'],cal['rotation'])
                intr=np.array(cal['camera_intrinsic'])*.64;intr[2,2]=1
                f={'camera_to_world':c2w.tolist(),'_w2c':np.linalg.inv(c2w),'intrinsics_1024':intr.tolist()}
                p=projection(a,f)
                if p is None:reasons['behind_or_near_plane']+=1;continue
                x0,y0,x1,y1=p['box'];w=x1-x0;h=y1-y0;area=w*h/(1024*576)
                if w<72 or h<40 or not .006<=area<=.18 or min(x0,y0,1024-x1,576-y1)<8:
                    reasons['size_border']+=1;continue
                path=factory/'rgb'/s['filename'];seen.add((tok,s['filename']))
                r={'source_id':sid,'instance_token':tok,'scene':c['scene'],'camera':channel,'keyframe_index':k,
                    'timestamp_us':s['timestamp'],'filename':s['filename'],'available':path.is_file(),
                    'box_xyxy':p['box'].tolist(),'view_angles_deg':view_angles(a,f).tolist(),
                    'camera_to_world':c2w.tolist(),'intrinsics_1024':intr.tolist(),'actor':a,
                    'source_mask_status':q.get('donor_mask_status'),'view_quality_status':'unreviewed',
                    'instance_mask_available':False,'purpose':'真实外部观测/选择供体视角；不是生成视角，也不自动送入DriveEditor'}
                if path.is_file():
                    im=np.asarray(Image.open(path).convert('RGB').resize((1024,576),Image.Resampling.LANCZOS));b=np.array([x0-12,y0-12,x1+12,y1+12]).astype(int);b[[0,2]]=b[[0,2]].clip(0,1024);b[[1,3]]=b[[1,3]].clip(0,576)
                    crop=im[b[1]:b[3],b[0]:b[2]];r['sharpness_hint']=float(cv2.Laplacian(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var())
                    dest=f'{sid}_{channel}_{k}.jpg';Image.fromarray(crop).save(out/'crops'/dest,quality=94);r['crop']='crops/'+dest
                else:missing.append(s['filename'])
                rows.append(r)
    bytok={}
    for r in rows:bytok.setdefault(r['instance_token'],[]).append(r)
    instances=[]
    for tok,rs in bytok.items():
        avail=[r for r in rs if r['available']]
        span=max((abs(wrap(a['view_angles_deg'][0]-b['view_angles_deg'][0])) for a in avail for b in avail),default=0)
        instances.append({'instance_token':tok,'source_ids':sorted({r['source_id'] for r in rs}),'available_views':len(avail),
                          'available_cameras':sorted({r['camera'] for r in avail}),'yaw_span_deg':span,
                          'genuine_multiple_cameras':len({r['camera'] for r in avail})>=2})
    summary={'candidate_views':len(rows),'available_views':sum(r['available'] for r in rows),'missing_RGB':len(set(missing)),
        'instances':len(instances),'instances_multiple_cameras':sum(r['genuine_multiple_cameras'] for r in instances),
        'instances_yaw_span_ge15':sum(r['yaw_span_deg']>=15 for r in instances),'rejection_counts':dict(reasons),
        'interpretation':'跨相机需同一instance+真实相机/曝光；现有RGB不足的视图未冒充已具备。可见性GT与图像框只作提案，mask/清洁度需GPU后质检。各相机作为独立训练clip；没有多相机神经网络输入。'}
    dump(out/'catalog.json',{'summary':summary,'instances':instances,'views':rows});dump(out/'missing_rgb.json',sorted(set(missing)))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--factory',type=Path,required=True);a=p.parse_args();main(a.root,a.factory)
