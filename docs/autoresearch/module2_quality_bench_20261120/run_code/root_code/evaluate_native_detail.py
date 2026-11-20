"""Native-resolution frozen teacher/renderer quality bench; CPU evaluation only."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time,hashlib,gc
import numpy as np
from scipy import ndimage as ndi

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1');G=Q/'gpu_20261119';B=G/'quality_bench_20261120_r1'
sys.path.insert(0,str(B/'metrics'));sys.path.insert(0,str(B/'root_code'))
from road_detail_metrics import evaluate_detail,ResolutionSpec,TileLayout
from evaluate_saved_lowres import compact

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def reconstruct_box4(image,context,linear):
    h,w,c=image.shape
    assert h%4==0 and w%4==0 and c==3
    # Normalize legal-source mass before and after interpolation; foreign RGB cannot leak
    # across domain boundaries. All-full blocks equal the historical encoded-sRGB box mean.
    num=np.where(context[...,None],image,0.).reshape(h//4,4,w//4,4,3).mean((1,3))
    mass=context.astype(float).reshape(h//4,4,w//4,4).mean((1,3))
    if not linear:
        n=np.repeat(np.repeat(num,4,axis=0),4,axis=1)
        m=np.repeat(np.repeat(mass,4,axis=0),4,axis=1)
        return np.divide(n,m[...,None],out=np.zeros_like(n),where=m[...,None]>0),m
    yy,xx=np.indices((h,w),dtype=float)
    # Edge-replicated align-centres reconstruction; an explicitly labelled teacher control, never GS.
    coords=[(yy+.5)/4-.5,(xx+.5)/4-.5]
    n=np.stack([ndi.map_coordinates(num[...,c],coords,order=1,mode='nearest',prefilter=False) for c in range(3)],-1)
    m=ndi.map_coordinates(mass,coords,order=1,mode='nearest',prefilter=False)
    return np.divide(n,m[...,None],out=np.zeros_like(n),where=m[...,None]>0),m

def main():
    high=B/'highres';out=B/'native_detail_r1';out.mkdir(exist_ok=False)
    assert (B/'lowres_r1/completed.json').is_file()
    start=time.monotonic();records=[];bindings={}
    write(out/'started.json',dict(frames=[110,86],primary_cases=['M01','M11','M10'],
        diagnostics=['M11_noH','teacher_identity','teacher_box4_nearest','teacher_box4_linear'],
        stage_references_images_only=['initial','low_A200'],alpha=.95,tile_side=128,
        control_selection='fixed before this evaluation outputs; not image-specific fitted filters',optimizer_updates=0,native_calls=0))
    for raw in (110,86):
        targetpath=high/f'targets/{raw:03d}.npz';targets=load(targetpath);bindings[str(targetpath)]=sha(targetpath)
        packet_paths={a:high/f'rendered/{a}/{raw:03d}.npz' for a in ('M01','M11','M10','M11_noH')}
        packets={a:load(p) for a,p in packet_paths.items()}
        bindings.update({str(p):sha(p) for p in packet_paths.values()})
        alpha=packets['M11']['road_alpha'].squeeze();assert alpha.shape==(1280,1920)
        for a,p in packets.items():assert np.array_equal(p['road_alpha'].squeeze(),alpha),(raw,a,'common alpha')
        stable=alpha>=.95
        for region in ('C','S','A'):
            if region=='A':names=('M11','teacher_box4_linear')
            else:names=('M01','M11','M10','M11_noH','teacher_identity','teacher_box4_nearest','teacher_box4_linear')
            teacher=targets['T_'+region].transpose(1,2,0).astype(np.float64)
            score=targets[region+'_pre_alpha'].astype(bool);context=score&stable
            if not score.any():
                records.append(dict(raw=raw,region=region,status='no_legal_native_reference'));continue
            for name in names:
                control_mass=None
                if name=='teacher_identity':pred=teacher
                elif name.startswith('teacher_box4_'):pred,control_mass=reconstruct_box4(teacher,context,name.endswith('linear'))
                else:
                    rgb=packets[name]['road_rgb'].transpose(1,2,0).astype(np.float64)
                    pred=np.divide(rgb,alpha[...,None],out=np.zeros_like(rgb),where=stable[...,None])
                role='real_observation' if region=='A' else 'frozen_pseudo_teacher'
                rep=evaluate_detail(pred,teacher,context_mask=context,score_mask=score,
                    resolution=ResolutionSpec(1920,1280,'native_camera1920_straight_road_alpha95','srgb'),
                    tile_layout=TileLayout(128,128),teacher_role=role,color_space='stable_alpha_unpremult_srgb',
                    input_conditioning=dict(common_frozen_road_alpha_threshold=.95,maximum_RGB_perturbation_amplification=1/.95,
                        native_resolution_actual_camera=True,teacher_source='original frozen native teacher',
                        score_before_Gaussian_alpha=True,geometry_missing_not_silently_removed=True))
                rep.update(raw=raw,region=region,case=name,score_excluded_by_alpha95=int((score&~stable).sum()),
                    surface_hit_score_pixels=int((score&targets['surface_hit']).sum()),
                    source_valid_score_pixels=int((score&targets['source_material_valid']).sum()),
                    teacher_or_method='teacher_only_degradation_control_not_GS' if name.startswith('teacher_') else 'actual_native_basis_world_read' if name!='M10' else 'actual_native_GS_average',
                    teacher_control_source_mass=None if control_mass is None else dict(
                        full_score_source_mass_min=float(control_mass[score].min()),
                        zero_mass_score_pixels=int((score&(control_mass==0)).sum()),
                        partial_mass_context_pixels=int((context&(control_mass<1-1e-12)).sum()),
                        boundary_policy='box4 and reconstruction normalize legal context mass; exact historic RGB box mean where full source blocks are legal'),
                    lane_structure_status='N/A until verified real visible paint/reference structure is supplied')
                filename=f'{raw:03d}_{region}_{name}.json';write(out/filename,rep)
                records.append(dict(raw=raw,region=region,case=name,file=filename,
                    score_excluded_by_alpha95=rep['score_excluded_by_alpha95'],
                    surface_hit_score_pixels=rep['surface_hit_score_pixels'],source_valid_score_pixels=rep['source_valid_score_pixels'],
                    teacher_or_method=rep['teacher_or_method'],**compact(rep)))
                x=rep['scales'][0]['core'];print(json.dumps(dict(raw=raw,region=region,case=name,pixels=x['pixels'],
                    cos=x['HF_cosine'],energy=x['HF_energy_ratio'],HF=x['HF_MAE'])),flush=True)
                del rep,pred;gc.collect()
        del packets,targets;gc.collect()
    result=dict(schema='native_road_quality_bench.v2',rows=records,input_sha256=bindings,
        seconds=time.monotonic()-start,optimizer_updates=0,native_calls=0,GPU_operations=0,
        comparison='Compare methods ONLY at same frame/domain/space/scale. Native and480 pixel scales are distinct. Teacher-only box4 reconstruction is not a GS prediction.')
    write(out/'summary.json',result);write(out/'completed.json',dict(rows=len(records),seconds=result['seconds'],optimizer_updates=0,native_calls=0))
    print(json.dumps(dict(output=str(out),rows=len(records),seconds=result['seconds'])))

if __name__=='__main__':main()
