"""从持久重载后的原生端点做固定区域显示裁片，不参与优化或指标计算。"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image

G=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119')
T=G/'quality_bench_20261120_r1/highres/targets'
I=G/'native_capacity_20261120_r1/qa/ideal_world_control_r2'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgb(a):
    a=np.asarray(a)
    if a.shape[0]==3:a=a.transpose(1,2,0)
    return np.round(np.clip(a,0,1)*255).astype(np.uint8)
def roi(mask,shape):
    h,w=shape;ys,xs=np.where(mask)
    if len(xs):cx,cy=int(np.median(xs)),int(np.median(ys))
    else:cx,cy=w//2,int(h*.75)
    cw,ch=min(512,w),min(384,h);x=int(np.clip(cx-cw//2,0,w-cw));y=int(np.clip(cy-ch//2,0,h-ch))
    return x,y,x+cw,y+ch

def main(manifest,output):
    manifest,output=Path(manifest),Path(output)
    if output.exists():raise FileExistsError('preserve prior preview run')
    output.mkdir(parents=True)
    source=json.loads(manifest.read_text());records=source.get('records',source.get('packets',source.get('rows')))
    if records is None:raise ValueError('packet manifest requires records/packets/rows')
    frames={};bindings=[];rois={}
    for raw in [86,110,134,179,198]:
        target=T/f'{raw:03d}.npz';frame=dict(teachers={},methods={});frames[str(raw)]=frame;rois[raw]={}
        if target.exists():
            with np.load(target,allow_pickle=False)as z:
                for s in ['C','S']:
                    rois[raw][s]=roi(z[s+'_pre_alpha'],(1280,1920))
            bindings.append(dict(role='teacher',raw=raw,path=str(target),sha256=sha(target),alpha_applied=False))
        else:
            for s in ['C','S']:rois[raw][s]=roi(np.zeros((1280,1920),bool),(1280,1920))
    # Complete both crop labels after both source ROIs are known.
    for raw in [86,110]:
        with np.load(T/f'{raw:03d}.npz',allow_pickle=False)as z:
            for s in ['C','S']:
                frames[str(raw)]['teachers'][s]=write_views(output,f'{raw:03d}_teacher_{s}',rgb(z['T_'+s]),rois[raw])
    for record in records:
        raw=int(record['raw']);arm=record['arm'];path=Path(record.get('path',record.get('packet')))
        if raw not in rois:continue
        with np.load(path,allow_pickle=False)as z:
            if z['road_rgb'].shape!=(3,1280,1920):raise ValueError('native1920 packet required')
            methods=frames[str(raw)]['methods'];methods[arm]={}
            for view,key in [('road','road_rgb'),('joint','joint_rgb')]:
                methods[arm][view]=write_views(output,f'{raw:03d}_{arm}_{view}',rgb(z[key]),rois[raw])
        bindings.append(dict(role='public_native_endpoint',arm=arm,raw=raw,path=str(path),sha256=sha(path)))
    for raw in [86,110]:
        frame=frames[str(raw)];ip=I/f'{raw:03d}.npz'
        if not ip.exists()or 'IDEAL_ORACLE'not in frame['methods']:continue
        with np.load(ip,allow_pickle=False)as z:
            target=write_views(output,f'{raw:03d}_ideal_target_P',rgb(z['road_target_P'][1]),rois[raw])
        frame['ideal']=dict(target=target,initial=frame['methods']['F0']['road'],endpoint=frame['methods']['IDEAL_ORACLE']['road'])
        bindings.append(dict(role='ideal_premult_target',raw=raw,path=str(ip),sha256=sha(ip),double_alpha_applied=False))
    result=dict(schema='native_capacity_previews.v1',frames=frames,source_manifest=str(manifest),source_manifest_sha256=sha(manifest),
        source_bindings=bindings,ROI_xyxy={str(k):v for k,v in rois.items()},ROI_selection='median of fixed source pre-alpha pixels; diagnostics fixed lower-center crop; never prediction/error selected',
        native_crop_resampling='none; actual stored1920 packet pixels',overview='640px width display only; bicubic',teacher_alpha_applied=False,GPU_operations=0,optimizer_updates=0)
    (output/'assets.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),files=len(list(output.glob('*.png'))),frames=len(frames)),ensure_ascii=False))

def write_views(output,stem,im,rois):
    image=Image.fromarray(im);w,h=image.size;overview=stem+'_overview.png'
    image.resize((640,round(h*640/w)),Image.Resampling.BICUBIC).save(output/overview)
    result=dict(overview='previews/'+overview)
    for s,box in rois.items():
        name=stem+'_'+s+'_native.png';image.crop(box).save(output/name);result[s+'_crop']='previews/'+name
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.manifest,a.output)
