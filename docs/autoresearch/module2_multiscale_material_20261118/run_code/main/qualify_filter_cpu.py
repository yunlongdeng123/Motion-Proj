"""固定真实Σ、actual A0/observed H-mask的GH15对GH41数值资格；0GPU。"""
from pathlib import Path
import json
import time
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter
from road_chart_sampling import StructuredRoadChart,local_chart_plan,exact_center_uv
from check_chart_against_aabb_cpu import bilinear

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
BASE=C/'road_asset_20261116_r1/phase2/formal_r1/asset'
COVER=C/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'


def load(path,keys):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in keys}


def take(ids,n):
    ids=np.unique(np.asarray(ids,np.int64));return ids[np.linspace(0,len(ids)-1,min(n,len(ids)),dtype=np.int64)] if len(ids) else ids


def quantiles(a):
    return dict(zip(('min','median','p95','p99','max'),np.quantile(a,[0,.5,.95,.99,1]).tolist()))


def integrate(uv,weights,A0,valid,mH,Hfield):
    n,q=uv.shape[:2];v=bilinear(valid[...,None],uv).reshape(n,q)
    base=bilinear(A0*valid[...,None],uv).reshape(n,q,3)
    gate=bilinear(mH[...,None],uv).reshape(n,q)
    high=bilinear(Hfield,uv).reshape(n,q,3)
    mass=np.einsum('q,nq->n',weights,v)
    a=np.einsum('q,nqc->nc',weights,base)/np.maximum(mass[:,None],1e-12)
    b=(np.einsum('q,nqc->nc',weights,base+v[...,None]*gate[...,None]*high))/np.maximum(mass[:,None],1e-12)
    return np.clip(a,0,1),np.clip(b,0,1),mass


def main():
    begin=time.monotonic()
    atlas=load(BASE/'base_atlas.npz',['surface_world_xyz','grid_world_row','world_from_uv','known'])
    a=load(P/'cpu_initial_r1/A0.npz',['rgb_srgb','valid','source_effective','confidence_effective'])
    road=load(COVER/'road.npz',['active','normal_world','gaussian_sigma_m'])
    provenance=load(COVER/'provenance.npz',['world_keys','world_xyz'])
    marks=load(BASE/'markings.npz',['observed_white'])['observed_white'][atlas['grid_world_row']]
    observed=a['valid']&(a['source_effective']==1)&atlas['known']
    detail=maximum_filter((marks&observed).astype(float),size=7,mode='constant')
    mH=observed*a['confidence_effective']*(.25+.75*detail)
    yy,xx=np.mgrid[:mH.shape[0],:mH.shape[1]]
    raw=np.stack((np.cos(np.pi*xx+.37),np.cos(np.pi*(xx+yy)/np.sqrt(2)+.19),
                  np.sin(2*np.pi*xx/3+.11)*np.cos(2*np.pi*yy/3-.43)),axis=-1)*((mH>0)[...,None])
    Hfield=.5*.04*(raw-gaussian_filter(raw,sigma=(3,3,0),mode='nearest',truncate=3))
    active=np.flatnonzero(road['active']);g=atlas['grid_world_row'];uvall,_=exact_center_uv(provenance['world_keys'],g,active)
    selection={'uniform_active':take(active,128)}
    selection['observed_white']=take(g[marks&observed],64)
    boundary=maximum_filter(a['valid'],size=3)!=minimum_filter(a['valid'],size=3)
    selection['source_boundary']=take(g[boundary],64)
    Hboundary=(maximum_filter(mH,size=3)-minimum_filter(mH,size=3))>1e-8
    selection['H_mask_boundary']=take(g[Hboundary],64)
    tri=[]
    for rawid in [86,110]:
        z=load(C/'road_capacity_cpu_20261117_r1'/f'supervision_v4_candidate_r1/{rawid:03d}.npz',
               ['S_target_shadow','triangle_world_rows','triangle_weights'])
        target=z['S_target_shadow'];tri.extend(z['triangle_world_rows'][target][z['triangle_weights'][target]>1e-12].tolist())
    selection['S_anchor_association']=take(tri,128)
    rows=np.intersect1d(np.unique(np.concatenate(list(selection.values()))),active)
    index=np.searchsorted(active,rows)
    query15=np.load(P/'chart_plan_candidate_r1/sample_uv_f64.npy',mmap_mode='r')[index]
    plan15=local_chart_plan(atlas['surface_world_xyz'],g,provenance['world_keys'],rows,road['normal_world'][rows],
         road['gaussian_sigma_m'][rows],atlas['world_from_uv'],order=15,world_xyz=provenance['world_xyz'][rows])
    plan41=local_chart_plan(atlas['surface_world_xyz'],g,provenance['world_keys'],rows,road['normal_world'][rows],
         road['gaussian_sigma_m'][rows],atlas['world_from_uv'],order=41,world_xyz=provenance['world_xyz'][rows])
    world41=plan41['world_xyz'][:,None,:2]+np.einsum('nij,qj->nqi',plan41['chol_world_xy'],plan41['standard_nodes'])
    query41=StructuredRoadChart(atlas['surface_world_xyz'],atlas['world_from_uv']).query(world41)['uv']
    A15,H15,m15=integrate(query15,plan15['weights'],a['rgb_srgb'],a['valid'].astype(float),mH,Hfield)
    A41,H41,m41=integrate(query41,plan41['weights'],a['rgb_srgb'],a['valid'].astype(float),mH,Hfield)
    eligible=m41>=.9
    comparisons={}
    for name,v15,v41 in [('actual_A0',A15,A41),('actual_A0_plus_bounded_H',H15,H41)]:
        error=abs(v15-v41);e=error[eligible]
        comparisons[name]=dict(reference_qualified_rows=int(eligible.sum()),reference_rule='GH41 validmass>=0.9, selected before inspecting RGB errors',
             RGB_abs_quantiles=quantiles(e),row_max_abs_quantiles=quantiles(e.max(1)),
             p99_limit=1/255,max_limit=4/255,passed=bool(np.quantile(e,.99)<=1/255 and e.max()<=4/255),
             all_sample_error_quantiles=quantiles(error),low_mass_rows=int((~eligible).sum()))
    x7,w7=np.polynomial.hermite.hermgauss(7);w7/=np.sqrt(np.pi)
    x15,w15=np.polynomial.hermite.hermgauss(15);w15/=np.sqrt(np.pi)
    transfer=[]
    for sigma in np.unique(np.quantile(road['gaussian_sigma_m'][active,:2],[0,.5,1])):
        for wavelength in [.24,.12,.06,.04]:
            for angle in [0,np.pi/4]:
                t=2*np.pi*sigma/wavelength*np.array([np.cos(angle),np.sin(angle)])
                exact=np.exp(-.5*np.dot(t,t))
                v7=float(np.prod([np.sum(w7*np.cos(np.sqrt(2)*z*x7)) for z in t]))
                v15=float(np.prod([np.sum(w15*np.cos(np.sqrt(2)*z*x15)) for z in t]))
                transfer.append(dict(sigma=float(sigma),wavelength_m=wavelength,direction_degrees=angle*180/np.pi,
                                     analytic=float(exact),GH7=v7,GH15=v15,GH7_abs_error=abs(v7-exact),GH15_abs_error=abs(v15-exact)))
    analytic_max=max(r['GH15_abs_error'] for r in transfer)
    np.savez_compressed(P/'qa/filter_numerical_samples.npz',world_rows=rows,reference_qualified=eligible,
        GH15_A0=A15,GH41_A0=A41,GH15_A0_plus_H=H15,GH41_A0_plus_H=H41,GH15_validmass=m15,GH41_validmass=m41)
    result=dict(schema='multiscale_exact_chart_filter_CPU_qualification.v1',GPU_executed=False,native_calls=0,training_updates=0,
        role='numerical integration engineering check, not material truth, visual acceptance, or representation-limit certificate',
        source_A0=str(P/'cpu_initial_r1/A0.npz'),source_geometry=str(COVER),GH15_cache=str(P/'chart_plan_candidate_r1/receipt.json'),
        sampling=dict(selection='frozen input only; deterministic stratified active/mark/source-boundary/H-boundary/S-triangle association',
                      rows=len(rows),selection_counts={k:len(v) for k,v in selection.items()},not_global_error_bound=True),
        H_fixture=dict(raw='deterministic bounded axis/diagonal/2D near-Nyquist fields; observed-parameter mask before blur',
                       effective_H_abs_max=float(abs(Hfield).max()),bound=.04,blur_sigma_m=.06,
                       gate='source1 AND known confidence*(.25+.75*observed-white7support)',pseudo_H_gate_zero=True,
                       bilinear_mask_times_bilinear_H_inside_filter=True),
        reference='GH41 on same exact world triangle chart; independent NumPy field implementation',
        comparisons=comparisons,analytic_transfer=transfer,GH15_analytic_response_error_max=analytic_max,
        GH15_analytic_limit=1e-4,GH15_analytic_pass=bool(analytic_max<1e-4),
        validmass=dict(GH15=quantiles(m15),GH41=quantiles(m41),absolute_difference=quantiles(abs(m15-m41)),
                       GH41_low_mass=int((m41<.9).sum()),GH15_low_mass=int((m15<.9).sum()),
                       qualification_disagreement=int(((m15>=.9)!=(m41>=.9)).sum()),
                       lowmass_source_and_training_effect='must be separately reported over fixedS3131/full3187; never shrink denominator'),
        all_registered_numerical_checks_pass=bool(analytic_max<1e-4 and all(v['passed'] for v in comparisons.values())),
        formal_authorized=False,human_verdict=None,elapsed_seconds=time.monotonic()-begin)
    (P/'qa/filter_numerical_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(P/'qa/filter_numerical_qualification.json'),rows=len(rows),
         checks=result['all_registered_numerical_checks_pass'],comparisons=comparisons,
         analytic_error_max=analytic_max,validmass=result['validmass'],seconds=result['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
