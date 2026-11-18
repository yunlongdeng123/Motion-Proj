"""只读门1既有NPZ：965低覆盖像素的网格状态与真实原生贡献佐证。"""
from pathlib import Path
import argparse,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import binary_erosion


def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if keys is None or k in keys}
def read(path):return json.loads(Path(path).read_text())
def write(path,d):Path(path).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def preview(path,probe,full,photo,baseline,maximum):
    rgb=np.uint8(np.floor(np.clip(probe['baseline_joint_rgb'].transpose(1,2,0),0,1)*255+.5));overlay=rgb.astype(np.float32)
    edge=full&~binary_erosion(full);low=full&~photo
    overlay[photo]=.6*overlay[photo]+.4*np.array([60,220,150]);overlay[low]=.5*overlay[low]+.5*np.array([235,90,220]);overlay[edge]=[80,220,255]
    views=[np.uint8(overlay)]
    for alpha in [baseline,maximum]:
        colors=np.zeros_like(rgb);colors[full&(alpha<.05)]=[255,80,100];colors[full&(alpha>=.05)&(alpha<=.9)]=[255,190,45];colors[full&(alpha>.9)]=[55,205,145]
        views.append(colors)
    labels=['Cached joint RGB: green=photo S, purple=low S','Cached baseline S: red=gap, amber=thin, green=support','Cached max-endpoint S: identical fixed denominator']
    canvas=Image.new('RGB',(1464,364),(10,17,27));draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    except OSError:font=ImageFont.load_default()
    for i,(value,label) in enumerate(zip(views,labels)):
        left=8+i*488;draw.text((left,9),label,font=font,fill=(225,235,245));canvas.paste(Image.fromarray(value),(left,32))
    canvas.save(path,lossless=True,method=4)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU follow-up requires empty CUDA_VISIBLE_DEVICES')
    ap=argparse.ArgumentParser();ap.add_argument('--gate',type=Path,required=True);ap.add_argument('--supervision',type=Path,required=True)
    ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    road=load(args.asset/'road.npz',['active','minimum_model_depth','gaussian_sigma_m','road_opacity'])
    prior=load(args.asset/'provenance.npz',['surface_supported','legal_write','unknown','known','guarded_world_rows','coverage_requested','coverage_trusted'])
    scope=load(args.gate/'qualified_scope.npz');active=scope['active_world_rows'];world_n=len(road['active']);rows=[]
    if not np.array_equal(active,np.flatnonzero(road['active'])):raise ValueError('active topology changed')
    ratio=np.maximum(1,np.minimum(scope['absolute_sigma_upper_m'][:,0]/road['gaussian_sigma_m'][:,0],scope['absolute_sigma_upper_m'][:,1]/road['gaussian_sigma_m'][:,1]))
    rowflags=dict(K_guard=scope['conservative_K_guard'],source_near=scope['near_guard'],prior_world_guard=prior['guarded_world_rows'][active],
        no_surface_support=~prior['surface_supported'][active],illegal_or_material_unknown=(~prior['legal_write']|prior['unknown'])[active],
        coverage_request_missing=~prior['coverage_requested'][active],known_observed_prior=prior['known'][active],
        physical_editable=scope['physical_legal'],tangent_growth_available=scope['physical_legal']&(ratio[active]>1.000001))
    for raw in [86,110]:
        d=load(args.supervision/f'{raw:03d}.npz');f=load(args.gate/f'{raw:03d}_coverage_probe.npz');attr=load(args.gate/f'{raw:03d}_actual_contribution.npz')
        full=d['S_target_shadow'];photo=d['S_target_shadow_photo'];alpha=f['baseline_road_only_alpha'];amax=f['combined_max_road_only_alpha'];low=full&(alpha<=.9)
        tri=d['triangle_world_rows'][low];weight=d['triangle_weights'][low];use=weight>1e-7
        if (tri<0).any() or (tri>=world_n).any() or not np.allclose(weight.sum(1),1,atol=1e-6):raise ValueError('qualified S lost triangle world mapping')
        def anyv(v):return (np.asarray(v)[tri]&use).any(1)
        def allv(v):return (np.asarray(v)[tri]|~use).all(1)
        flags=dict(no_surface_support=anyv(~prior['surface_supported']),source_near_constraint=anyv(road['minimum_model_depth']>0),
            prior_world_guard=anyv(prior['guarded_world_rows']),r4_K_guard=anyv(scope['world_conservative_K_guard']),
            illegal_or_material_unknown_corner=anyv(~prior['legal_write']|prior['unknown']),
            inactive_any_corner=anyv(~road['active']),inactive_all_corners=allv(~road['active']),
            coverage_request_missing_any=anyv(~prior['coverage_requested']),coverage_request_missing_all=allv(~prior['coverage_requested']),
            all_corners_active=allv(road['active']),all_corners_trusted=allv(prior['coverage_trusted']),
            no_allowed_tangent_growth_on_all_corners=allv(ratio<=1.000001))
        exclusive={};remaining=np.ones(int(low.sum()),bool)
        for name in ['no_surface_support','source_near_constraint','prior_world_guard','r4_K_guard','illegal_or_material_unknown_corner','inactive_all_corners','inactive_any_corner','coverage_request_missing_any']:
            exclusive[name]=int((remaining&flags[name]).sum());remaining&=~flags[name]
        exclusive['active_supported_no_marked_constraint']=int(remaining.sum())
        zeros=alpha[low]==0;maxzeros=amax[low]==0
        flags['baseline_exact_zero']=zeros;flags['maximum_endpoint_exact_zero']=maxzeros
        cross={key:dict(pixels=int(mask.sum()),baseline_zero=int((mask&zeros).sum()),max_endpoint_zero=int((mask&maxzeros).sum())) for key,mask in flags.items()}
        native=[];names=attr['region_names'].tolist()
        for name in ['S','S_full','S_gap','S_thin']:
            mass=attr['alpha_mass'][:,names.index(name)].astype(np.float64);total=float(mass.sum());positive=mass>0
            region=photo if name=='S' else full if name=='S_full' else full&(alpha<.05) if name=='S_gap' else full&(alpha>=.05)&(alpha<=.9)
            expected=float(alpha[region].sum(dtype=np.float64))
            native.append(dict(region=name,pixels=int(region.sum()),actual_contributor_rows=int(positive.sum()),actual_alphaT_mass=total,
                float_alpha_region_sum=expected,relative_conservation_error=abs(total-expected)/max(expected,1),
                overlapping_row_properties={key:dict(contributor_rows=int((positive&mask).sum()),actual_mass=float(mass[mask].sum()),fraction=float(mass[mask].sum()/total) if total>0 else None) for key,mask in rowflags.items()}))
        if any(x['relative_conservation_error']>5e-4 for x in native):raise ValueError('actual native attribution is inconsistent with cached alpha')
        row=dict(raw=raw,full_S=int(full.sum()),photo_S=int(photo.sum()),fixed_low_S=int(low.sum()),maximum_endpoint_reaches_support=int((low&(amax>.9)).sum()),
            baseline_gap=int((low&(alpha<.05)).sum()),baseline_thin=int((low&(alpha>=.05)).sum()),maximum_endpoint_gap=int((low&(amax<.05)).sum()),
            triangle_property_cross_table=cross,triangle_properties_are_overlapping=True,exclusive_priority_triangle_classes=exclusive,
            actual_native_region_contribution=native,
            causal_boundary='triangle node status is a local source-state association, not per-pixel Gaussian contribution attribution; native mass is exact per region but is not localized to individual low pixels',
            max_endpoint_is_not_global_supremum_certificate=True,
            endpoint_alpha_decrease_gt_1e_5_pixels=int((low&(amax<alpha-1e-5)).sum()))
        rows.append(row);maps={key:np.zeros_like(low) for key in flags}
        for key,mask in flags.items():maps[key][low]=mask
        np.savez_compressed(args.output/f'{raw:03d}_source_state_maps.npz',fixed_low_S=low,**maps)
        preview(args.output/f'{raw:03d}_cached_S_alpha_diagnostic.webp',f,full,photo,alpha,amax)
        print(json.dumps(dict(raw=raw,fixed_low_S=row['fixed_low_S'],property_counts={k:v['pixels'] for k,v in cross.items()},exclusive=exclusive),ensure_ascii=False),flush=True)
    if sum(x['fixed_low_S'] for x in rows)!=965:raise ValueError('fixed low-coverage denominator changed')
    write(args.output/'remaining_S_source_and_native_mass.json',dict(schema='road_capacity_remaining_S_CPU_audit.v1',rows=rows,fixed_low_S_total=965,
        source_asset=str(args.asset),source_gate=str(args.gate),native_scope_eligible_rows=int(scope['eligible_color'].sum()),
        active_topology_or_geometry_changed=False,CPU_only=True,new_GPU_operations=0,new_optimizer_updates=0,new_scene_render_calls=0,
        visualization='two lossless WebP diagnostics compose ONLY existing native_r1 arrays and frozen masks; no new scene image is generated by GPU',
        conclusion='existing road query mesh reaches S, but old Gaussian activation/request support is mismatched with the new target-shadow domain; most low S triangle anchors are inactive. Real native low-mass evidence and exact-zero pixels establish a coverage floor, not a larger-2D-generator or color-capacity failure.',
        scope_limit='no activation/surface/scale/opacity experiment is started; mother protocol keeps active/topology fixed',
        maximum_endpoint_note='max permitted parameter endpoint showed no support gains; no global all-parameter-state optimum claim is made',
        initial_engineering_retry=dict(stage='first output path already existed; error before native source load',native_forwards=0,optimizer_updates=0,retained=True)))


if __name__=='__main__':main()
