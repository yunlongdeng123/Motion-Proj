"""Retain an early-stopped actual joint render as an explicit failed control."""
from pathlib import Path
import argparse
import numpy as np
from local_gaussian_asset import bank,read,write
from render_new_local_zero import pin,png


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['run','zero','masks','contract','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=read(a.run/'receipt.json');cfg=read(a.contract)
    assert not r['complete'] and r['nativeF']==2 and r['nativeB']==0
    assert not a.output.exists();a.output.mkdir(parents=True)
    raw=110;path=a.run/'110_failed_RGBA.npz';z=bank(path);ref=bank(a.zero/'110/joint_render.npz')
    m=bank(a.masks/'110/scope_masks.npz');ev=bank(Path(cfg['fixed_eval'])/'110/fixed_eval.npz')
    rgb=z['rgb'];alpha=z['alpha'];d=abs(rgb-ref['rgb']).max(0);ad=abs(alpha-ref['alpha'])
    record=dict(raw=110,source=pin(path),failed_actual_render_not_completed_control=True,domains={})
    for name in ['full_image','outside_allowed_edit','outside_replacement','actor_protected','K_protected']:
        mask=m[name];record['domains'][name]=dict(pixels=int(mask.sum()),RGB_changed_exact=int((mask&(d>0)).sum()),
            RGB_changed_above1e_6=int((mask&(d>1e-6)).sum()),alpha_changed_exact=int((mask&(ad>0)).sum()),
            max_RGB=float(d[mask].max(initial=0)),max_alpha=float(ad[mask].max(initial=0)))
    full=ev['full_original_band'];holes=full&(alpha<.95)&(ref['alpha']>=.95)
    record.update(full_band_new_alpha95_holes_vs_v4zero=int(holes.sum()),unknown_new_alpha95_holes=int((holes&~ev['ground_eval']).sum()),
        fixed_original_full_band_pixels=int(full.sum()),unknown_denominator=int((full&~ev['ground_eval']).sum()))
    for name,condition in [('red_diagnostic',(rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)),
        ('blue_cyan_diagnostic',(rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08))]:
        mask=ev[name];record[name]=dict(fixed_pixels=int(mask.sum()),remaining=int((mask&condition).sum()))
    png(a.output/'110_failed_joint.png',rgb);png(a.output/'110_alpha.png',alpha)
    record['joint_PNG']=pin(a.output/'110_failed_joint.png');record['alpha_PNG']=pin(a.output/'110_alpha.png')
    write(a.output/'receipt.json',dict(complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        actual_failed_run=pin(a.run/'receipt.json'),quality_accepted=False,actual_retirement_RGBA_invariance=False,rows=[record]))
    print(record)

if __name__=='__main__':main()
