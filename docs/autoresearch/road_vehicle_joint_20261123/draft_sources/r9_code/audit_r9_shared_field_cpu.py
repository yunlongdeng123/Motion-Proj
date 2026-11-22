"""Independent fixed-domain metrics and full-surface views; never per-view C/S/W."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,binary_erosion
from PIL import Image,ImageDraw


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def pin(p):
    p=Path(p);return {'path':p.resolve().as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def stats(a):
    a=np.asarray(a);return {'count':int(a.size),'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),
      'p95':float(np.quantile(a,.95)),'p99':float(np.quantile(a,.99)),'max':float(a.max())} if a.size else {'count':0}
def lf(rgb,mask,sigma):
    den=gaussian_filter(mask.astype(np.float64),sigma)
    num=gaussian_filter(np.where(mask[...,None],rgb,0).astype(np.float64),(sigma,sigma,0))
    return np.divide(num,den[...,None],out=np.zeros_like(num),where=den[...,None]>1e-10)
def srgb(x):
    x=np.clip(x,0,1);return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)
def png(p,x):Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(p)
def gather(a,uv):
    h,w=a.shape[:2];q=np.nan_to_num(uv.astype(np.float64));x=np.clip(q[...,0],0,w-1);y=np.clip(q[...,1],0,h-1)
    ix=np.floor(x).astype(int);iy=np.floor(y).astype(int);jx=np.minimum(ix+1,w-1);jy=np.minimum(iy+1,h-1)
    dx=x-ix;dy=y-iy;out=0
    for yy,xx,v in [(iy,ix,(1-dx)*(1-dy)),(iy,jx,dx*(1-dy)),(jy,ix,(1-dx)*dy),(jy,jx,dx*dy)]:
        out=out+a[yy,xx]*(v[...,None] if a.ndim==3 else v)
    valid=np.isfinite(uv).all(-1)&(uv[...,0]>=0)&(uv[...,1]>=0)&(uv[...,0]<=w-1)&(uv[...,1]<=h-1)
    return out,valid,(iy,ix,jy,jx)
def all4(a,index):
    iy,ix,jy,jx=index;return a[iy,ix]&a[iy,jx]&a[jy,ix]&a[jy,jx]
def any4(a,index):
    iy,ix,jy,jx=index;return a[iy,ix]|a[iy,jx]|a[jy,ix]|a[jy,jx]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work-root',type=Path,default=Path(__file__).resolve().parents[2]);a=ap.parse_args()
    work=a.work_root.resolve();r9=work/'old_raw_teacher_joint_20261123_r9';r7=work/'road_shared_appearance_20261122_r7'
    out=r9/'cpu_review/joint_shared_v1';out.mkdir(parents=True,exist_ok=True)
    contract_path=r9/'acceptance_contract_v1.json';contract=read(contract_path);producer=read(r9/'joint_shared_v1/receipt.json')
    fp=r9/'joint_shared_v1/final_atlas.npz';sp=r9/'joint_shared_v1/solve_state.npz';seedp=r7/'quilt_seed/atlas_seed.npz'
    f=load(fp);state=load(sp);seed=load(seedp);g=load(r7/'shared_grid_geometry.npz');real=load(r7/'real_observation_atlas.npz')
    rgb=f['atlas_linear'];base=seed['atlas_seed_linear'];kind=f['source_kind'];known=seed['same_world_lock'];perm=seed['write_domain'];adopt=f['teacher_adopted']
    valid=state['valid'];raw=state['raw_RGB'].astype(np.float64);corrected=state['corrected_RGB'].astype(np.float64)
    checks={
      'contract_SHA':pin(contract_path)['sha256']==producer['contract_sha256'],
      'field_SHA':pin(fp)['sha256']==producer['final_atlas_sha256'],
      'state_SHA':pin(sp)['sha256']==producer['solve_state_sha256'],
      'seed_SHA':pin(seedp)['sha256']==producer['seed_sha256'],
      'real_6407':int(known.sum())==6407,
      'real_RGB_byteexact_seed':rgb[known].tobytes()==base[known].tobytes(),
      'real_RGB_byteexact_actual_observation':rgb[known].tobytes()==real['canonical_linear_rgb'][known].tobytes(),
      'outside_permission_RGB_byteexact':rgb[~perm].tobytes()==base[~perm].tobytes(),
      'permission_exact_seed':bool(np.array_equal(f['permission_mask'],perm)),
      'real_locks_exact_seed':bool(np.array_equal(f['real_locks'],known)),
      'source1_exact_real':bool(np.array_equal(kind==1,known)),
      'all_RGB_finite':bool(np.isfinite(rgb).all()),'linear_RGB_range':bool((rgb>=0).all() and (rgb<=1).all()),
      'permission_unknown_zero':not bool(((kind==0)&perm).any()),
      'teacher_support_exact':bool(np.array_equal(f['teacher_support_count'],valid.sum(0))),
      'teacher_adoption_exact':bool(np.array_equal(adopt,(state['teacher_alpha']>1e-8)&perm&~known)),
      'kind3_exact_adoption':bool(np.array_equal(kind==3,adopt))}
    pairs=[];sigma=contract['joint_solver']['shared_tone_sigma_m']/.02
    low_raw=[lf(x,m,sigma) for x,m in zip(raw,valid)];low_corrected=[lf(x,m,sigma) for x,m in zip(corrected,valid)]
    for i in range(len(raw)):
      for j in range(i+1,len(raw)):
        m=valid[i]&valid[j];b=float(abs(raw[i]-raw[j])[m].mean());c=float(abs(corrected[i]-corrected[j])[m].mean())
        l0=float(abs(low_raw[i]-low_raw[j])[m].mean());l1=float(abs(low_corrected[i]-low_corrected[j])[m].mean())
        pairs.append(dict(raw_frames=[int(state['frames'][i]),int(state['frames'][j])],fixed_pair_texels=int(m.sum()),
          fixed_channel_samples=int(m.sum())*3,full_RGB_before=b,full_RGB_after=c,full_RGB_improvement=1-c/b,
          LF_sigma_m=sigma*.02,LF_before=l0,LF_after=l1,LF_improvement=1-l1/l0,
          LF_pass=bool(l1<=.005 or l1<=.8*l0),real_overlap_texels=int((m&known).sum()),
          robust_zero_weight_samples_not_removed=int((m&((state['robust_weights'][i]<=0)|(state['robust_weights'][j]<=0))).sum())))
    checks['all_fixed_pairs_LF_contract_pass']=all(p['LF_pass'] for p in pairs)
    checks['fixed_denominator_matches_producer']=all(p['fixed_channel_samples']==q['fixed_channel_samples'] for p,q in zip(pairs,producer['fixed_pair_metrics']))
    # Replay robust-mean passes independently. Saved robust_weights are the NEXT
    # weights after the third mean, not the generating weights of the saved mean.
    weights=state['base_weights'].astype(np.float64);rw=weights.copy();shared=np.zeros_like(base,dtype=np.float64)
    for _ in range(3):
      den=rw.sum(0);shared=np.divide((corrected*rw[...,None]).sum(0),den[...,None],out=np.zeros_like(shared),where=den[...,None]>1e-12)
      residual=np.sqrt(((corrected-shared[None])**2).mean(-1));rw=weights*np.minimum(1,.01/np.maximum(residual,1e-12))
    alpha=state['teacher_alpha'].astype(np.float64)
    replay=((1-alpha[...,None])*base+alpha[...,None]*shared).astype(np.float32);replay[known]=base[known];replay[~perm]=base[~perm];replay=np.clip(replay,0,1)
    replay_error=float(abs(replay-rgb).max());checks['full_RGB_fusion_replay_with_float32_export_tolerance']=replay_error<=2e-7
    # Fixed LF boundary test: all horizontal/vertical source3 <-> non-source3
    # edges with BOTH endpoints inside original permission and material support.
    material=perm&(kind>0);final_lf=lf(rgb,material,sigma);seed_lf=lf(base,material,sigma)
    boundary_values=[];seed_boundary_values=[];boundary_rows=[]
    for axis in [0,1]:
      x=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1));y=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
      edge=material[x]&material[y]&(adopt[x]!=adopt[y])
      v=np.abs(final_lf[x]-final_lf[y])[edge].ravel();b=np.abs(seed_lf[x]-seed_lf[y])[edge].ravel()
      boundary_values.append(v);seed_boundary_values.append(b);boundary_rows.append({'axis':axis,'edges':int(edge.sum()),'final_LF_jump':stats(v),'seed_same_edge_LF_jump':stats(b)})
    boundary=stats(np.concatenate(boundary_values));baseline_boundary=stats(np.concatenate(seed_boundary_values))
    checks['boundary_LF_contract_pass']=bool(boundary['p95']<=.005 or boundary['p95']<=baseline_boundary['p95']+.0025)
    # Confidence is producer metadata and not calibrated; it is not used here.
    confidence={str(k):stats(f['confidence'][(kind==k)&perm]) for k in np.unique(kind[perm])}
    views=[];panels=[];gains={int(x['raw']):float(x['gain']) for x in read(r7/'exposure_gain.json')['records']};rim=None
    colours=np.array([[.12,.15,.18],[.12,.85,.85],[.77,.50,.18],[.68,.35,.95]])
    for frame in [86,110,134,179,198]:
      cp=r7/f'projection_cache_v1/{frame:03d}/view_cache.npz';c=load(cp);uv=c['atlas_uv_map'];geom=c['geometry_valid']&c['projection_valid']
      projected,bounds,idx=gather(rgb,uv);seed_projected,_,_=gather(base,uv)
      permitted=all4(perm,idx);anyperm=any4(perm,idx);available=all4((kind>0)&np.isfinite(rgb).all(-1),idx)
      evaluated=geom&bounds&permitted&available
      no_geometry=~geom;uv_unsupported=geom&~bounds;outside=geom&bounds&~anyperm
      border=geom&bounds&anyperm&~permitted;unknown=geom&bounds&permitted&~available
      yy,xx=np.indices(geom.shape);hatch=((xx+yy)%8)<3
      final_image=np.full((*geom.shape,3),[.0627,.0902,.1333]);seed_image=final_image.copy()
      final_image[evaluated]=srgb(projected[evaluated]/gains[frame]);seed_image[evaluated]=srgb(seed_projected[evaluated]/gains[frame])
      png(out/f'{frame:03d}_material_only.png',final_image);png(out/f'{frame:03d}_seed_material_only.png',seed_image)
      final_hatch=final_image.copy();seed_hatch=seed_image.copy()
      for m,col in [(no_geometry,[.28,.30,.33]),(uv_unsupported,[.8,.1,.7]),(outside,[.36,.34,.24]),(border,[.95,.6,.05]),(unknown,[1,.1,.15])]:
        final_hatch[m&hatch]=col;seed_hatch[m&hatch]=col
      png(out/f'{frame:03d}_material_hatched.png',final_hatch);png(out/f'{frame:03d}_seed_material_hatched.png',seed_hatch)
      source_colour,_,_=gather(colours[kind],uv);source_image=np.full_like(final_image,[.0627,.0902,.1333]);source_image[evaluated]=source_colour[evaluated]
      png(out/f'{frame:03d}_source_kind.png',source_image)
      np.savez_compressed(out/f'{frame:03d}_domain_masks.npz',geometry=geom,evaluated=evaluated,unsupported_geometry=no_geometry,
        UV_unsupported=uv_unsupported,outside_permission=outside,permission_border=border,authorized_unknown=unknown)
      # Optional photo context is separately named, never a supervision tensor.
      context=c['photo_pinhole_srgb'].copy();context[evaluated]=final_image[evaluated]
      png(out/f'{frame:03d}_optional_photo_context.png',context)
      row=dict(raw=frame,projection_cache=pin(cp),source_geometry_fixed_denominator=int(geom.sum()),
        full_image_pixel_count=int(geom.size),evaluated_material=int(evaluated.sum()),outside_permission=int(outside.sum()),
        bilinear_permission_border=int(border.sum()),authorized_unknown=int(unknown.sum()),UV_unsupported=int(uv_unsupported.sum()),
        old_CS_W_used_for_scope=False,photo_pixels_used_as_material=False,exposure_gain=gains[frame],
        evaluated_material_finite=bool(np.isfinite(projected[evaluated]).all()))
      views.append(row);write(out/f'{frame:03d}_scope.json',row)
      # Contact display crop is fixed, while full native pinhole PNGs above retain all pixels.
      bbox=(0,400,1152,768);left=Image.open(out/f'{frame:03d}_seed_material_hatched.png').crop(bbox).resize((864,276))
      right=Image.open(out/f'{frame:03d}_material_hatched.png').crop(bbox).resize((864,276))
      panel=Image.new('RGB',(1728,306),'#111827');draw=ImageDraw.Draw(panel)
      draw.text((8,8),f'{frame:03d}  SEED  |  display crop only; full-res PNG preserved',fill='white')
      draw.text((872,8),f'{frame:03d}  JOINT RAW FIELD  |  same actual-surface support',fill='white')
      panel.paste(left,(0,30));panel.paste(right,(864,30));panels.append(panel)
      if frame==110:
        mp=r7/'cpu_review/black_rim_domain_diagnostic_v1/fixed_primary1792_masks.npz';fixed=load(mp)['rim1792'];assert fixed.sum()==1792
        partitions={'covered_new_field_RGB':fixed&evaluated,'unsupported_existing_geometry':fixed&no_geometry,
          'unsupported_UV':fixed&uv_unsupported,'outside_permission':fixed&outside,'permission_bilinear_border':fixed&border,'authorized_material_unknown':fixed&unknown}
        union=np.zeros_like(fixed)
        for v in partitions.values():assert not (v&union).any();union|=v
        assert np.array_equal(union,fixed)
        np.savez_compressed(out/'fixed1792_classification.npz',fixed1792=fixed,**partitions)
        rim={'fixed_mask':pin(mp),'fixed_denominator':1792,'counts':{k:int(v.sum()) for k,v in partitions.items()},
          'partition_complete':True,'covered_encoded_RGB_max_below_40_255':int((partitions['covered_new_field_RGB']&(final_image.max(-1)<40/255)).sum()),
          'covered_new_linear_RGB':stats(projected[partitions['covered_new_field_RGB']].ravel()),
          'unsupported_pixels_are_not_teacher_black':True,'original_photo_not_used_to_fill_new_material':True,
          'geometry_or_alpha_fixed_by_this_CPU_atlas':False}
    sheet=Image.new('RGB',(1728,sum(p.height for p in panels)));top=0
    for p in panels:sheet.paste(p,(0,top));top+=p.height
    sheet.save(out/'five_view_seed_final_material_contactsheet.png')
    atlas_panels=[]
    for name,x in [('SEED',base),('JOINT RAW FIELD',rgb)]:
      img=srgb(x);img[~material]=[.06,.09,.13];im=Image.fromarray(np.rint(img*255).astype(np.uint8));panel=Image.new('RGB',(im.width,im.height+24),'#111827');ImageDraw.Draw(panel).text((8,5),name,fill='white');panel.paste(im,(0,24));atlas_panels.append(panel)
    img=Image.new('RGB',(atlas_panels[0].width,sum(p.height for p in atlas_panels)));top=0
    for p in atlas_panels:img.paste(p,(0,top));top+=p.height
    img.save(out/'world_atlas_seed_final.png');png(out/'world_source_kind.png',colours[kind]);png(out/'world_teacher_alpha.png',alpha)
    checks['fixed1792_partition_complete']=rim['partition_complete']
    checks['all_view_authorized_unknown_zero']=all(x['authorized_unknown']==0 for x in views)
    result={'schema':'R9.independent_shared_field_metrics.v1','complete':True,
      'decision':'LOCAL_NUMERIC_CHECKS_PASS' if all(checks.values()) else 'FAIL',
      'final_atlas':pin(fp),'solve_state':pin(sp),'contract':pin(contract_path),'checks':checks,'fixed_pairs':pairs,
      'boundary_definition':'All adjacent source3/non-source3 edges with both endpoints in original permission/material; sigma .12m normalized LF; same edges for seed baseline',
      'boundary_axes':boundary_rows,'boundary_LF_jump':boundary,'seed_boundary_LF_jump':baseline_boundary,
      'full_RGB_field_replay_max_abs_with_float32_state':replay_error,
      'robust_weights_metadata_role':'Post-third-mean NEXT IRLS weights, not exact weights generating the saved mean; independent replay uses three mean/weight passes.',
      'confidence_by_source_kind':confidence,'confidence_is_calibrated_probability':False,'confidence_used_in_independent_QA':False,
      'source_kind_names':{'0':'unknown / outside source support','1':'same-world observed real photo','2':'similar real-photo quilt inferred','3':'jointly corrected original raw teacher full RGB + seed boundary blend'},
      'views':views,'fixed1792':rim,'contactsheet':pin(out/'five_view_seed_final_material_contactsheet.png'),
      'world_atlas_preview':pin(out/'world_atlas_seed_final.png'),
      'source_sampling_qualification_replay':'PENDING separate independent audit',
      'visual_acceptance':'PENDING independent inspection and root review',
      'full_scene_restoration_claim':False,'CPU_only':True,'GPU_forward':0,'native_F':0,'native_B':0,'diffusion_calls':0,'Gaussian_updates':0}
    write(out/'independent_metrics.json',result)
    print(json.dumps({'decision':result['decision'],'checks':checks,'fixed_pairs':pairs,'boundary_LF_p95':boundary['p95'],
      'baseline_boundary_LF_p95':baseline_boundary['p95'],'fixed1792':rim,'receipt':pin(out/'independent_metrics.json')}))


if __name__=='__main__':main()
