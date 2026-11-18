"""Recompute all frozen five-arm metrics and directional source decompositions, CPU only."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from .material_evaluation import evaluate_arm,load,read,write

FRAMES=(86,110,134,179,198)
def directional(P,f,mask=None):
    V=f['stripe_vertical'].copy();H=f['stripe_horizontal'].copy()
    if mask is not None:
        V&=mask[:-2]&mask[1:-1]&mask[2:];H&=mask[:,:-2]&mask[:,1:-1]&mask[:,2:]
    dy=np.abs(P[:,2:]-2*P[:,1:-1]+P[:,:-2]);dx=np.abs(P[:,:,2:]-2*P[:,:,1:-1]+P[:,:,:-2])
    ey=float(dy[:,V].mean())if V.any()else None;ex=float(dx[:,H].mean())if H.any()else None
    return dict(E_vertical=ey,E_horizontal=ex,ratio_vertical_over_horizontal=ey/ex if ey is not None and ex else None,V_stencils=int(V.sum()),H_stencils=int(H.sum()))

def collect(gpu,out):
    gpu=Path(gpu);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    manifest=gpu/'evaluation/frozen_r2/manifest.json';m=read(manifest);F={r:load(m['frames'][str(r)]['path'])for r in FRAMES}
    arms=['M00','M10','M01','M11','T11'];others=['pixel_initial','pixel_common_A200','avg_initial','avg_common_A200']
    dirs={a:gpu/f'formal_r1/{a}/C800'for a in arms};dirs.update({a:gpu/f'formal_r1/{a}'for a in others})
    results={};packets={}
    for a,p in dirs.items():
        paths={r:p/f'rendered/{r:03d}.npz'for r in FRAMES};reload=next(iter(p.glob('*reload*.json')),None)
        if a=='M11':
            bound=gpu/'field_training/persistence_179_r1/receipt.json'
            if bound.exists():reload=bound
        results[a]=evaluate_arm(paths,manifest,out/a,arm_id=a,stage='C800'if a in arms else a,public_reload=reload)
        packets[a]={r:load(x)for r,x in paths.items()}
        print(a,results[a]['mechanical_pass'],flush=True)
    components=[]
    for r in FRAMES:
        f=F[r];b=load(gpu/f'native/basis/{r:03d}/road_basis.npz')
        pieces={'native_parent':f['baseline_road_rgb'],'native_readonly':b['direct_readonly_rgb'],
            'native_eligible':b['eligible_rgb']}
        pieces.update({a:packets[a][r]['road_rgb']for a in dirs})
        for a in arms:pieces[a+' minus initial']=packets[a][r]['road_rgb']-packets['avg_initial'if a in ['M00','M10']else'pixel_initial'][r]['road_rgb']
        pieces['M10 minus M00']=packets['M10'][r]['road_rgb']-packets['M00'][r]['road_rgb']
        pieces['M11 minus M01']=packets['M11'][r]['road_rgb']-packets['M01'][r]['road_rgb']
        components.append(dict(raw=r,role='teacher_source'if r in [86,110]else'diagnostic_no_GT',components={k:directional(v.astype(np.float64),f)for k,v in pieces.items()}))
    teacher=[]
    for r in [86,110]:
        f=F[r]
        for region in ['A','C','S']:
            target=f['baseline_road_alpha'][None]*f['target_'+region]
            teacher.append(dict(raw=r,region=region,teacher_directional=directional(target.astype(np.float64),f,f[region]),
                actual_directional={a:directional(packets[a][r]['road_rgb'].astype(np.float64),f,f[region])for a in arms}))
    summary={}
    for a,v in results.items():
        summary[a]=dict(mechanical_pass=v['mechanical_pass'],photo={k:v['pooled'][k]['MAE']for k in ['A','C','S','S_new','S_thin']},
            HF1={k:v['detail'][k]['1px']for k in ['A','C','S','S_new']},edges={k:v['detail'][k]['edges']for k in ['A','C','S','S_new']},
            stripe=[dict(raw=r['raw'],**r['stripe'])for r in v['rows']],crossview=v.get('crossview'))
    write(out/'summary.json',summary)
    write(out/'directional_source_decomposition.json',dict(schema='fixed_stencil_CPU_source_decomposition.v1',rows=components,teacher=teacher,
        interpretation='Directional second-difference energy is not a semantic stripe classifier. Initial A0, native readonly and eligible pieces are shown; absolute energies are not additive and give no causal fraction. Learned-minus-initial and matched permission differences diagnose added energy. No teacher for diagnostic views.',CPU_native_calls=0))
    print(str(out/'summary.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-dir',required=True);p.add_argument('--out',required=True);a=p.parse_args();collect(a.gpu_dir,a.out)
