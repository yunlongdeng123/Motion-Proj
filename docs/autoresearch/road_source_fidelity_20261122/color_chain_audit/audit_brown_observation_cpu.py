"""固定已存 native 输出，对照 raw/Brown-rectified 观测；无新渲染、无拟合。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,gaussian_filter,minimum_filter


def metrics(observed,rendered,mask):
    if not mask.any():return dict(n=0)
    rows=[]
    for sigma in (1.2,5/3,3.):
        a=(observed-gaussian_filter(observed,(sigma,sigma,0)))[mask].astype(float).ravel()
        b=(rendered-gaussian_filter(rendered,(sigma,sigma,0)))[mask].astype(float).ravel()
        an,bn=np.linalg.norm(a),np.linalg.norm(b)
        rows.append(dict(sigma=sigma,HF_cosine=float(a@b/(an*bn)) if an*bn else None,
                         observation_HF_RMS=float(np.sqrt(np.mean(a*a))),
                         render_HF_RMS=float(np.sqrt(np.mean(b*b)))))
    return dict(n=int(mask.sum()),RGB_MAE=float(abs(observed-rendered)[mask].mean()),
                mean_render_minus_photo=(rendered-observed)[mask].mean(0).tolist(),spectral=rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_source_fidelity_20261122_r2'))
    ap.add_argument('--bank',type=Path,default=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs'))
    args=ap.parse_args();started=time.monotonic();rows=[]
    plan=json.loads((args.root/'probe_plan.json').read_text());plans={(p['candidate_id'],int(p['raw'])):p for p in plan['records']}
    for run in ('fixed_color_operator_02_115','fixed_color_operator_00_137'):
        receipt=json.loads((args.root/run/'receipt.json').read_text())
        r=receipt['rows'][0];patch=r['patch']
        for e in r['evaluations']:
            raw=e['raw'];p=args.root/run/patch/f'{raw:03d}'/'source_float.npz'
            with np.load(p,allow_pickle=False) as z:f={k:z[k] for k in z.files}
            obs=json.loads((args.bank/f'observations/{raw:03d}.json').read_text())
            photo=np.asarray(Image.open(args.bank/obs['image_path']).convert('RGB'),np.float32)/255
            x0,y0,x1,y1=f['bbox_xyxy'];yy,xx=np.mgrid[y0:y1,x0:x1]
            fx,fy,cx,cy,k1,k2,p1,p2,k3=obs['camera_intrinsic_Brown']
            x=(xx+.5-cx)/fx;y=(yy+.5-cy)/fy;r2=x*x+y*y
            rad=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
            sx=fx*(x*rad+2*p1*x*y+p2*(r2+2*x*x))+cx-.5
            sy=fy*(y*rad+p1*(r2+2*y*y)+2*p2*x*y)+cy-.5
            rectified=np.stack([map_coordinates(photo[:,:,c],[sy,sx],order=1,mode='nearest',prefilter=False) for c in range(3)],-1)
            inside=(sx>=0)&(sx<=photo.shape[1]-1)&(sy>=0)&(sy<=photo.shape[0]-1)
            record=plans[(patch,raw)]
            legal=np.asarray(Image.open(record['core_photo_valid_mask_full1920_remote_stage']))>0
            legal_rect=map_coordinates(legal.astype(np.uint8),[sy,sx],order=0,mode='constant',cval=0)>0
            common=f['score_mask']&inside&legal_rect
            strict=common&(minimum_filter(common.astype(np.uint8),size=15,mode='constant',cval=0)>0)
            assert np.array_equal(photo[y0:y1,x0:x1],f['observation'])
            data=dict(run=run,patch=patch,raw=raw,role=e['role'],score_original_n=int(f['score_mask'].sum()),
                common_n=int(common.sum()),strict_sigma1p667_support_n=int(strict.sum()),
                crop_map_displacement_L2=dict(median=float(np.median(np.hypot(sx-xx,sy-yy))),
                                            max=float(np.max(np.hypot(sx-xx,sy-yy)))),comparisons={})
            for key in ('original_rgb','direct_rgb','solved_rgb'):
                data['comparisons'][key]=dict(raw=metrics(f['observation'],f[key],common),
                    Brown_rectified=metrics(rectified,f[key],common),
                    raw_strict=metrics(f['observation'],f[key],strict),
                    Brown_rectified_strict=metrics(rectified,f[key],strict))
            rows.append(data)
    print(json.dumps(dict(schema='fixed_native_vs_Brown_rectified_observation_cpu.v1',new_native_renders=0,
        optimizer_updates=0,assets_modified=False,observation_remap_only=True,rows=rows,
        map_contract='Output native pixel(x,y) -> ((x+.5-cx)/fx,(y+.5-cy)/fy) -> Brown -> raw input(fx*xd+cx-.5,fy*yd+cy-.5). Bilinear encoded RGB; nearest legal mask.',
        evaluation_contract='Paired raw and rectified metrics use identical original score intersect remapped photo-legal and valid-input masks. No alignment search or color refit.',
        seconds=time.monotonic()-started),ensure_ascii=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
