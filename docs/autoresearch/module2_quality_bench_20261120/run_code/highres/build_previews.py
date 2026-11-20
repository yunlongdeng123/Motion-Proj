"""Small lossless native crops and explicitly downsampled overviews; CPU only."""
from pathlib import Path
import argparse,json,hashlib,datetime
import numpy as np
from PIL import Image

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1');G=Q/'gpu_20261119'
DEFAULT=G/'quality_bench_20261120_r1/highres'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}
def bbox(mask,pad=32):
    y,x=np.where(mask)
    return [max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(1920,int(x.max())+pad+1),min(1280,int(y.max())+pad+1)]
def rgb8(a):return np.clip(np.moveaxis(a,0,-1)*255+.5,0,255).astype(np.uint8)
def main(root):
    root=Path(root);out=root/'previews';out.mkdir(exist_ok=False);ROIs=[]
    old=read(G/'evaluation/frozen_r2/rois.json')['rois']
    for raw in [110,86]:
        t=load(root/f'targets/{raw:03d}.npz',['C_pre_alpha','S_pre_alpha'])
        for role in ['C','S']:
            ROIs.append(dict(raw=raw,id=role+'_native_source',role=role,bbox_xyxy=bbox(t[role+'_pre_alpha']),
                based_on='fixed native '+role+'_pre_alpha mask bbox +32px; no predicted RGB used',preview=True))
        for r in old:
            if r['raw']==raw and r['key']in ['C','S_full']:
                ROIs.append(dict(raw=raw,id=r['key']+'_historical',role=r['key'],bbox_xyxy=[4*x for x in r['bbox_xyxy']],
                    based_on='unchanged historical480 ROI coordinates multiplied by4; coordinates only',preview=False))
    roi_meta=dict(schema='highres_preview_ROIs_mask_only.v1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        created_after_GPU_outputs=True,prediction_selected=False,ROIs=ROIs,
        preview_policy='Native source ROIs emitted. Historical ROIs preserved as coordinates only to limit takeback, not replaced or scored differently.')
    write(out/'roi_manifest.json',roi_meta) # Persist before any predicted RGB is opened.
    assets=[];dedup={}
    def emit(name,a,meta):
        assert a.ndim==3 and a.shape[0]==3
        image=rgb8(a);key=hashlib.sha256(np.asarray(image.shape,np.int64).tobytes()+image.tobytes()).hexdigest()
        if key in dedup:dest=dedup[key]
        else:
            dest=name+'.webp';Image.fromarray(image).save(out/dest,format='WEBP',lossless=True,method=3);dedup[key]=dest
        assets.append(dict(file=dest,shape_HWC=list(image.shape),byte_size=(out/dest).stat().st_size,**meta))
    for raw in [110,86]:
        sources=[('T_A',root/f'targets/{raw:03d}.npz','T_A','原始观测（车下无GT）'),
            ('T_C',root/f'targets/{raw:03d}.npz','T_C','固定孔洞教师T（伪真值，未乘alpha）'),
            ('T_S',root/f'targets/{raw:03d}.npz','T_S','固定去影教师T（伪真值，未乘alpha）')]
        for case in ['M01','M11','M10','M11_noH']:sources.append((case+'_road_P',root/f'rendered/{case}/{raw:03d}.npz','road_rgb',case+'道路层P（预乘）'))
        sources.extend([('M11_joint_J',root/f'rendered/M11/{raw:03d}.npz','joint_rgb','M11原生联合J'),
            ('static_contribution',root/f'native_r1/{raw:03d}/joint_basis.npz','static_rgb','实际静态分支RGB贡献（不是去静态重渲染）')])
        for role,p,key,label in sources:
            a=load(p,[key])[key];assert a.shape==(3,1280,1920)
            overview=a.reshape(3,320,4,480,4).mean((2,4))
            emit(f'{raw:03d}_{role}_overview',overview,dict(raw=raw,role=role,label=label,kind='overview_only',
                source=str(p),source_key=key,source_resolution=[1920,1280],display_resolution=[480,320],
                resampling='float encoded-sRGB exact4x4 mean from actual native input; overview not used for scoring',alpha_applied_to_teacher=False))
            for r in ROIs:
                if r['raw']!=raw or not r['preview']:continue
                x0,y0,x1,y1=r['bbox_xyxy'];crop=a[:,y0:y1,x0:x1]
                emit(f'{raw:03d}_{r["id"]}_{role}',crop,dict(raw=raw,role=role,label=label,kind='native_1to1_crop',ROI=r['id'],bbox_xyxy=r['bbox_xyxy'],
                    source=str(p),source_key=key,source_resolution=[1920,1280],resampling='none; original native pixels; lossless WebP8 display only',alpha_applied_to_teacher=False))
            del a
    files={p.name:dict(bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest())for p in out.glob('*.webp')}
    total=sum(v['bytes']for v in files.values())
    write(out/'assets_manifest.json',dict(schema='highres_native_preview_assets.v1',assets=assets,files=files,total_unique_WebP_bytes=total,
        takeback_budget_bytes=5*1024**2,budget_pass=total<=5*1024**2,teacher_never_alpha_multiplied=True,
        float_authority='Original native float packets and original teacher source remain remote; display8bit quantization is not metric input',
        source_attribution='static_contribution is native same-sort RGB contribution, never a remove-static counterfactual',
        GPU_operations=0,new_native_calls=0,optimizer_updates=0))
    print(json.dumps(dict(path=str(out),unique_files=len(files),asset_roles=len(assets),total_bytes=total,budget_pass=total<=5*1024**2)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();main(a.root)
