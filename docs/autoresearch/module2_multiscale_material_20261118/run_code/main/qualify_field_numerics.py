"""读结果前固定的真实chart/GH15数值门；无native、无优化/参数搜索。"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from world_material_field import WorldMaterialField
from road_chart_sampling import make_exact_sample_plan
from prepare_real_field import BASE,COVER,load

def summary(error,mass):
    groups={}
    for name,mask in [('mass_ge_point9',mass>=.9),('mass_lt_point9',mass<.9)]:
        e=error[mask]
        groups[name]=dict(rows=int(mask.sum()),RGB_error_p50=float(np.quantile(e,.5)) if len(e) else None,
            RGB_error_p99=float(np.quantile(e,.99)) if len(e) else None,RGB_error_max=float(e.max()) if len(e) else None)
    return groups

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    a=load(BASE/'base_atlas.npz');r=load(COVER/'road.npz');p=load(COVER/'provenance.npz',['world_xyz','world_keys','B32_new','effective_material_available'])
    rows=np.flatnonzero(r['active']);g=a['grid_world_row'];valid=a['legal']&~a['unknown']&(a['source']>0)
    A0=a['rgb_srgb'].copy();source=a['source'].copy();conf=a['confidence'].copy();patch=p['B32_new'][g]
    A0[patch]=r['reference_srgb'][g[patch]];source[patch]=3;valid[patch]=True;conf[patch]=.25;A0[~valid]=0
    observed=(source==1)&a['known']&valid
    white=load(BASE/'markings.npz',['observed_white'])['observed_white'][g]&observed
    mH=observed*conf*(.25+.75*maximum_filter(white.astype(float),size=7,mode='constant'))
    generator=np.random.default_rng(101)
    with np.load('/root/autodl-tmp/drivingeditor_repro/campaign/road_activation_20261118_r1/native_r1/asset_A1_guarded/road.npz') as f:parent_active=f['active']
    strata=[rows,rows[r['gaussian_sigma_m'][rows,0]>=.01954],rows[white.ravel()[np.argsort(g.ravel())][rows]],rows[~parent_active[rows]]]
    chosen=np.unique(np.concatenate([generator.choice(s,min(1024,len(s)),replace=False) for s in strata if len(s)]))
    # 固定seed及规则，既有32pseudo和全部new892也纳入；不是学习结果筛选。
    zero=load(BASE/'road.npz',['active'])['active'];chosen=np.unique(np.r_[chosen,np.flatnonzero(r['active']&~zero)])
    model=[];receipts=[]
    for order in [15,41]:
        plan=make_exact_sample_plan(a['surface_world_xyz'],g,p['world_keys'],chosen,r['normal_world'][chosen],
            r['gaussian_sigma_m'][chosen],a['world_from_uv'],order=order,world_xyz=p['world_xyz'][chosen])
        receipts.append(plan['sampling_receipt']);model.append(WorldMaterialField(A0,valid,mH,plan))
    tests=[]
    H,W=valid.shape;y,x=np.mgrid[:H,:W]
    patterns=[('A0_only',None),('H_Nyquist_u',np.cos(np.pi*x)),('H_period6cm_diagonal',np.cos(2*np.pi*(x+y)/(3*np.sqrt(2)))),
              ('H_observed_white_edge',np.where(white,1.,-1.))]
    for name,pattern in patterns:
        values=[];masses=[]
        for m in model:
            with torch.no_grad():
                m.high.zero_()
                if pattern is not None:m.high.copy_(torch.as_tensor(np.arctanh(.9)*np.repeat(pattern[...,None],3,2),dtype=m.high.dtype))
                color,mass=m.query(chunk=256)
            values.append(color.numpy());masses.append(mass.numpy())
        error=np.abs(values[0]-values[1]).max(1)
        tests.append(dict(case=name,**summary(error,masses[1])))
        np.savez_compressed(output/(name+'.npz'),world_rows=chosen,GH15=values[0],dense_GH41=values[1],kernel_valid_mass=masses[1],RGB_max_channel_error=error)
    frequency=[]
    for angle in [0,22.5,45,67.5,90]:
        k=2*np.pi/.04*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        n,w=np.polynomial.hermite.hermgauss(15);xx,yy=np.meshgrid(n,n);wx,wy=np.meshgrid(w,w)
        q=np.stack([xx.ravel(),yy.ravel()],1)*np.sqrt(2)*.0195499996
        prediction=float(((wx*wy).ravel()/np.pi*np.cos(q@k)).sum());truth=float(np.exp(-.5*.0195499996**2*np.dot(k,k)))
        frequency.append(dict(angle_degrees=angle,analytic=truth,GH15=prediction,absolute_error=abs(prediction-truth)))
    passed=max(v['absolute_error'] for v in frequency)<1e-4 and all(
        t['mass_ge_point9']['RGB_error_p99']<=1/255 and t['mass_ge_point9']['RGB_error_max']<=4/255 for t in tests)
    result=dict(schema='multiscale_numeric_qualification.v1',CPU_only=True,native_calls=0,optimizer_updates=0,
        chosen_rows=len(chosen),choice_rule='seed101 up to1024 each all-active/max-sigma/observed-white/newactivation plus allnew892',
        sampled_world_rows_file='A0_only.npz',frequency=frequency,tests=tests,chart=receipts,
        numeric_gate_pass=passed,kernel_candidate='GH15',dense_reference='GH41 exactworldchart',
        thresholds=dict(frequency_max=1e-4,mass_ge_point9_RGB_p99=1/255,mass_ge_point9_RGB_max=4/255))
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output)
