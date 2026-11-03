"""Build a compact 3D repair envelope; native markings retain authority."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time,argparse
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_closing,binary_dilation,binary_opening,distance_transform_edt,gaussian_filter,label
import r4_runtime as r4
from editor_common import cpu,save

RUN=Path(__file__).resolve().parent.parent
OLD=RUN.with_name('shadow_a5_a37_199_20261103_r4')
r4.RUN=OLD;r4.INPUT=OLD/'inputs'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='inputs');args=parser.parse_args()
    start=time.time();dest=RUN/args.output_name;dest.mkdir(exist_ok=False);(dest/'views').mkdir()
    settings=json.loads((RUN/'quality_settings.json').read_text())
    coverage=settings['coverage'];rgb=settings['adaptive_rgb'];mark=settings['markings']
    assert coverage['preserve_observed_shadow_core'] and not mark['protect_white_native_ghosts']
    assert coverage['longitudinal_outer_m']>0 and coverage['lateral_outer_m']>0 and 0<=coverage['core_fraction']<1
    e=r4.Experiment();r=e.ref;old_active=int(r['active'].sum());N=len(r['active'])
    stats=RUN/'diagnostic/surface_stats.npz'
    if not stats.exists():stats=RUN.with_name('shadow_quality_20261103_r5')/'diagnostic/surface_stats.npz'
    s=np.load(stats)
    keys=r['keys'];ij=keys-keys.min(0);shape=tuple(ij.max(0)+1)
    def chart(x):
        a=np.zeros(shape,dtype=x.dtype);a[ij[:,0],ij[:,1]]=x;return a
    def gather(x):return x[ij[:,0],ij[:,1]]
    mark_count=np.zeros(N);valid_count=np.zeros(N);shadow_count=np.zeros(N);masks={}
    for raw in range(89,138):
        d=e.view(raw);uid=d['unit_id'];domain=d['repair_mask']&(uid>=0)
        b=cpu(e.base(raw)).transpose(1,2,0);bl=cpu(r4.linear(e.base(raw))).transpose(1,2,0)
        yellow=(b[:,:,0]>1.22*b[:,:,2])&(b[:,:,1]>1.08*b[:,:,2])&(b[:,:,0]>.32)
        # White residual car paint cannot serve as a road marking veto.
        components,num=label(yellow);sizes=np.bincount(components.ravel());sizes[0]=0
        marking=yellow&(sizes[components]>=mark['connected_yellow_min_pixels'])&(d['actor_labels']==0)&domain
        mark_count+=np.bincount(uid[marking],minlength=N);valid_count+=np.bincount(uid[domain],minlength=N)
        observed_shadow=d['shadow_proxy']&domain
        shadow_count+=np.bincount(uid[observed_shadow],minlength=N)
        masks[raw]=binary_dilation(marking,iterations=1)
    fraction=s['dark']/np.maximum(s['count'],1)
    # Original observed-shadow witnesses are BUILD evidence, including cells
    # where the native artifact is now bright rather than multiplicatively dark.
    mandatory=r['active']&(shadow_count>0)
    seed=r['active']&(((fraction>.18)&(s['dark']>=2))|mandatory)
    seedmap=binary_closing(chart(seed),iterations=2)
    seedmap=binary_opening(seedmap,iterations=1)
    seedmap|=chart(mandatory)
    # A 16cm support halo and 24cm smooth falloff are represented by Gaussian
    # opacities, never by copying pixels between independently rendered images.
    dist=distance_transform_edt(~seedmap,sampling=(.08/coverage['longitudinal_outer_m'],.08/coverage['lateral_outer_m']))
    envelope=np.clip((1-dist)/(1-coverage['core_fraction']),0,1)
    marked=(mark_count/np.maximum(valid_count,1)>mark['world_vote_fraction'])&(mark_count>=2)
    markedmap=binary_dilation(chart(marked),iterations=1)&~binary_dilation(chart(mandatory),iterations=1)
    envelope[markedmap]=0
    weight=gather(envelope).astype(np.float32)*r['active']
    active=r['active']&(weight>.005)
    # Low-frequency native radiance calibration uses only clean non-actor
    # measurements and is capped at 0.012 linear RGB per channel.
    sigma=(rgb['sigma_longitudinal_m']/.08,rgb['sigma_lateral_m']/.08)
    cc=chart(s['clean_count']);norm=gaussian_filter(cc.astype(float),sigma)
    correction=np.stack([gather(gaussian_filter(chart(s['residual_sum'][:,ch]),sigma)/np.maximum(norm,1e-8)) for ch in range(3)],1)
    correction=np.clip(correction,-rgb['max_linear_delta'],rgb['max_linear_delta'])*rgb['strength']*float(rgb['enabled'])
    r['active']=active;r['edit_weight']=weight;r['marking_cells']=marked;r['native_correction_linear']=correction
    r['reference_linear']=np.clip(r['reference_linear']+correction,0,1)
    r['reference_srgb']=np.where(r['reference_linear']<=.0031308,12.92*r['reference_linear'],1.055*np.maximum(r['reference_linear'],1e-10)**(1/2.4)-.055)
    # Compensate four-way overlap when tapering splat opacity.
    r['road_opacity']=np.where(weight>.999,.99005,1-(1-weight)**.40).astype(np.float32)
    np.savez_compressed(dest/'road_reference.npz',**r)
    for raw in range(89,138):
        d=e.view(raw).copy();uid=d['unit_id'];safe=np.maximum(uid,0)
        domain=d['repair_mask']&(uid>=0);w=weight[safe]*domain
        d['edit_weight']=w;d['marking_mask']=masks[raw]
        d['fit_weight']=w*np.load(OLD/'inputs/road_reference.npz')['fit_weight'][safe]
        d['recovery_reference_linear']=r['reference_linear'][safe]
        d['native_keep_weight']=np.where(d['protect_mask'],1.,np.maximum(1-w,masks[raw].astype(float))).astype(np.float32)
        np.savez_compressed(dest/'views'/('%03d.npz'%raw),**d)
    # Candidate row identities are inherited; all are regularized toward native
    # and the loss protects native RGB throughout the clean road domain.
    import shutil
    shutil.copy2(OLD/'inputs/contributors.npz',dest/'contributors.npz')
    rec=dict(source_frames=199,updates=0,old_active_cells=old_active,active_cells=int(active.sum()),
        marking_cells=int(marked.sum()),observed_shadow_cells=int(mandatory.sum()),observed_shadow_core_coverage=float((weight[mandatory]>.99).mean()),core_cells=int((weight>.99).sum()),taper_cells=int(((weight>.005)&(weight<.99)).sum()),
        parent_unchanged=e.unchanged(),seconds=time.time()-start,seed_rule='native luminance deficit>0.025 and ratio<.78 in >18% observed pixels, chroma<.6; >=2 dark samples',
        envelope='3D chart closing/opening, 0.16m halo +0.24m taper; connected yellow marking exclusion; original observed-shadow BUILD witnesses retain full coverage',
        calibration='clean native minus real199 atlas; spatial sigma0.32m; capped0.012 linearRGB',
        no_hard_image_route=True,only_road_time_stable=True,settings=settings)
    r4.write(dest/'receipt.json',rec);print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
