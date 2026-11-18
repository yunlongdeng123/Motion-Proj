"""小型真实几何/冻结材质密度2x数值对照；无native/GPU/优化器。"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.ndimage import gaussian_filter
from pixel_quadrature_stream import axis_counts,one_pixel_samples

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=B/'road_teacher_detail_cpu_20261119_r1';P=B/'road_multiscale_material_20261118_r1'


def bilinear(field,uv):
    if field.ndim==2:field=field[...,None]
    h,w,c=field.shape;lo=np.floor(uv).astype(int);t=uv-lo;out=np.zeros((len(uv),c))
    for dx,dy in[(0,0),(1,0),(0,1),(1,1)]:
        x,y=lo[:,0]+dx,lo[:,1]+dy;valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        f=(t[:,0]if dx else 1-t[:,0])*(t[:,1]if dy else 1-t[:,1])
        out[valid]+=field[y[valid],x[valid]]*f[valid,None]
    return out


def query(uv,base,valid,gate,H):
    v=bilinear(valid,uv)[:,0];mass=float(v.mean())
    b=bilinear(base*valid[...,None],uv)
    color=(b+v[:,None]*bilinear(gate,uv)*bilinear(H,uv)).mean(0)/max(mass,1e-12)
    return np.clip(color,0,1),mass


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--density',type=float,default=2.)
    parser.add_argument('--output',type=Path,default=Q/'pixel_render/quadrature_numeric_r1');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('保留旧数值现场，换输出目录')
    start=time.monotonic()
    with np.load(Q/'permission/final_r1/teacher_evidence.npz')as z:
        base=z['common_base_srgb'].astype(float);valid=z['common_material_valid'].astype(float);obs=z['observed_H_prior'].astype(float);union=z['mH_union'].astype(float);latent=z['common_latent_mask'].astype(float)
    bounds=json.loads((Q/'protocol/bound_recipe.json').read_text());high_bound=bounds['high_bound']
    h,w=valid.shape;yy,xx=np.indices((h,w));raw=.8*np.sin(2*np.pi*xx/4)*np.cos(2*np.pi*yy/6)*latent
    H0=.5*high_bound*(raw-gaussian_filter(raw,3.,mode='nearest',truncate=3.));H=np.stack((H0,.8*H0,-.6*H0),-1)
    rows=[];colors=[];metadata=[]
    for frame in(86,110):
        with np.load(Q/f'pixel_render/maps/{frame:03d}.npz')as m,np.load(P/f'stage0_native_r1/{frame:03d}_targets.npz')as t:
            C=m['center_uv'].reshape(-1,2);J=m['pixel_uv_jacobian'].reshape(-1,2,2);sizes=axis_counts(J,args.density);cost=sizes.prod(1)
            for domain in('A','C','S'):
                ids=np.flatnonzero(t[domain]);ordered=ids[np.argsort(cost[ids],kind='stable')]
                chosen=ordered[np.unique(np.rint(np.linspace(0,len(ordered)-1,8)).astype(int))]
                for pixel in chosen:
                    uv0,_,s0=one_pixel_samples(C[pixel],J[pixel],density=args.density);uv1,_,s1=one_pixel_samples(C[pixel],J[pixel],density=2*args.density)
                    for name,g,Hfield in[('base',union,np.zeros_like(H)),('observed_H_stress',obs,H),('teacher_H_stress',union,H)]:
                        a,m0=query(uv0,base,valid,g,Hfield);b,m1=query(uv1,base,valid,g,Hfield)
                        rows.append(dict(raw=frame,domain=domain,pixel=int(pixel),case=name,coarse_points=s0['points'],dense_points=s1['points'],
                            row_max=float(abs(a-b).max()),coarse_mass=m0,dense_mass=m1));colors.append(np.stack((a,b)))
    result=dict(schema='actual_pixel_box_dense2x_CPU_numerics.v1',density=args.density,dense_density=2*args.density,
        rows=rows,selected_pixels=len(rows)//3,RGB_p99=float(np.percentile([r['row_max']for r in rows],99)),RGB_max=max(r['row_max']for r in rows),
        base_p99=float(np.percentile([r['row_max']for r in rows if r['case']=='base'],99)),
        CPU_seconds=time.monotonic()-start,common_high_bound=high_bound,GPU_executed=False,native_calls=0,optimizer_updates=0,
        field='actual frozen commonbase + deterministic permitted-grid Hstress, no teacher imageRGB imported',
        sampling_target='same local canonical-UV pixelbox; nx/ny separate, denominator globally percompletepixel',
        no_domain_shrunk=True,convergence_proxy_only='dense2x is reference refinement, not strict quadrature certificate')
    args.output.mkdir(parents=True);np.savez_compressed(args.output/'colors.npz',coarse_dense=np.array(colors))
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('rows',)}))


if __name__=='__main__':main()
