"""Compare controls only after audit_blue.py's original-input mask registry is sealed."""
from pathlib import Path
import argparse,json
import numpy as np
from audit_blue import group_stats,sha,BBOX
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--parent',type=Path,default=BASE/'layers_128/r0_delete.npz');args=ap.parse_args()
    registry=json.loads((BASE/'roi_registration.json').read_text(encoding='utf8'))
    masks=np.load(BASE/'fixed_masks_raw128.npz')
    obs=np.load(BASE/'layers_128/observations.npz');target=obs['rgb'].transpose(2,0,1)
    variants={'parent':np.load(args.parent),'candidate':np.load(args.candidate)}
    out={'name':args.name,'mask_registry_sha256':sha(BASE/'roi_registration.json'),
         'mask_npz_sha256':sha(BASE/'fixed_masks_raw128.npz'),'candidate_sha256':sha(args.candidate),'parent_sha256':sha(args.parent),
         'evaluation_masks_unchanged':True,'regions':{},'matched_tangent_residual_jumps':{},
         'limits':['Same-scene development comparison; raw128 original observed road only',
                   'Band ground mapping interpolated from real sparse raw returns, not dense physical GT',
                   'A zero-support far edge is unsupported, never replaced by a better-looking boundary',
                   'A colour fit can compensate residual static blending; does not certify ordering correctness']}
    domains={key:masks[key] for key in masks.files if masks[key].dtype==bool}
    # These full-image original-observation masks are the existing parent protection definitions.
    domains.update(global_observed_road=obs['road'].astype(bool),
                   global_protected=obs['protect'].astype(bool)&~obs['visible'].any(0))
    out['geometry_weight_exact_comparison']={key:bool(np.array_equal(variants['parent'][key],variants['candidate'][key]))
        for key in ['alpha','group_alpha_T','newroad_alpha_T','oldroad_alpha_T','depth_m']}
    for key,m in domains.items():
        if m.dtype!=bool:continue
        region={}
        for name,d in variants.items():
            s=group_stats(d,m)
            if m.any():
                r=d['rgb'][:,m]-target[:,m]
                change=d['rgb'][:,m]-variants['parent']['rgb'][:,m]
                s.update(RGB_abs_error_sum=float(abs(r).mean(0).sum()),RGB_error_denominator_pixels=int(m.sum()),
                         RGB_abs_error_mean=float(abs(r).mean()),mean_signed_residual_rgb=r.mean(1).astype(float).tolist(),
                         versus_parent_RGB_abs_change_mean=float(abs(change).mean()),versus_parent_RGB_abs_change_max=float(abs(change).max()))
            region[name]=s
        out['regions'][key]=region
    xyz=masks['local_xyz']
    for edge,meta in registry['edge_centres'].items():
        minus=masks[f'band_{edge}_minus'];plus=masks[f'band_{edge}_plus'];tan=xyz[...,meta['tangent_axis']]
        # Fixed .2m bins, counts >=3 each side, based only on frozen geometry/masks.
        bins=np.floor(tan/.2).astype(np.int32)
        pairs=[]
        for b in np.unique(bins[minus|plus]):
            mm=minus&(bins==b);mp=plus&(bins==b)
            if mm.sum()>=3 and mp.sum()>=3:pairs.append((int(b),mm,mp))
        values={}
        for name,d in variants.items():
            rows=[]
            for b,mm,mp in pairs:
                resid=d['rgb']-target
                jump=resid[:,mp].mean(1)-resid[:,mm].mean(1)
                rows.append({'tangent_bin':b,'minus_count':int(mm.sum()),'plus_count':int(mp.sum()),
                             'signed_residual_jump_rgb':jump.astype(float).tolist(),
                             'mean_abs_channel_jump':float(abs(jump).mean())})
            values[name]={'paired_bins':len(rows),'equal_bin_abs_residual_jump_mean':float(np.mean([r['mean_abs_channel_jump'] for r in rows])) if rows else None,'bins':rows}
        out['matched_tangent_residual_jumps'][edge]=values
    path=BASE/f'{args.name}_fixed_blue_comparison.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    x0,y0,x1,y1=BBOX
    fig,ax=plt.subplots(2,3,figsize=(9,7),layout='constrained')
    for j,(title,img) in enumerate([('Original',obs['rgb']),('Parent',variants['parent']['rgb'].transpose(1,2,0)),(args.name,variants['candidate']['rgb'].transpose(1,2,0))]):
        ax[0,j].imshow(img[y0:y1,x0:x1],extent=[x0,x1,y1,y0]);ax[0,j].set_title(title)
    for j,key in enumerate(['newroad_alpha_T','oldroad_alpha_T','group_alpha_T']):
        a=variants['candidate'][key]-variants['parent'][key]
        if key=='group_alpha_T':a=a[3]
        ax[1,j].imshow(a[y0:y1,x0:x1],cmap='coolwarm',vmin=-.2,vmax=.2,extent=[x0,x1,y1,y0]);ax[1,j].set_title(['delta newroad alphaT','delta oldroad alphaT','delta static alphaT'][j])
    fig.suptitle('Fixed blue bbox; quantitative road/guard/world-band masks saved separately')
    fig.savefig(BASE/f'{args.name}_fixed_blue_comparison.png',dpi=170);plt.close(fig)
    print(json.dumps({'path':str(path),'original_road':{n:{k:s[k] for k in ['RGB_abs_error_mean','pixels']} for n,s in out['regions']['original_road'].items()},'jumps':out['matched_tangent_residual_jumps']},ensure_ascii=False))

if __name__=='__main__':main()
