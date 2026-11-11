"""零更新检查候选交接的真实贡献与原参考不变性。"""
from pathlib import Path
import argparse,json
import numpy as np
def main():
    import torch
    from PIL import Image
    from drivingeditor_dynamic_edit.v44_factory import from_manifest
    p=argparse.ArgumentParser();p.add_argument('--before',required=True,type=Path);p.add_argument('--after',required=True,type=Path);p.add_argument('--target-track',required=True);a=p.parse_args()
    out=a.after/'result_verification';out.mkdir(exist_ok=False)
    scene,m=from_manifest(a.after/'final_manifest.json');deleted=m.actor_ids([a.target_track]);N=len(scene.static.get_xyz);device=scene.static.get_xyz.device
    old=np.load(a.before/'gate1000/contributors.npz')['ids'];new=np.load(a.after/'gate1000/contributors.npz')['ids'];added=np.setdiff1d(new,old)
    source={r['frame_key']:r for r in json.loads((a.after/'delete_white/provenance.json').read_text())['frames']};records=[]
    with torch.no_grad():
        for frame,data in zip(m.frames,m.data['frames']):
            if frame.key not in [60,119,150,198]:continue
            key=frame.key;values,owners,cam,_=scene._rows(key,deleted,(),True);f=values['features'].clone();f[:,8:11]=0
            f[torch.as_tensor(new,device=device),8]=1;f[torch.as_tensor(added,device=device),9]=1;f[:N,10]=1
            pack=scene._bindings().raster(cam,scene.pipe,0,dict(values,features=f,mask=values['mask']&~torch.isin(owners,owners.new_tensor(deleted))),scene._background)
            rgb=pack['rgb']+(1-pack['alpha'])*scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            z=np.load(a.after/'delete_white'/source[key]['files']['float_npz']);err=float(abs(rgb.cpu().numpy()-z['rgb']).max());assert err<3e-6
            target=np.isin(np.asarray(Image.open(data['instance_mask']).resize((cam.image_width,cam.image_height),Image.Resampling.NEAREST)),deleted)
            with np.load(a.after/'world_road/views'/f'{key:03d}.npz') as d:regions=dict(target=target,fit=target&(d['fit']>.05),outside=target&~d['inside'],unknown=target&d['unknown_target'])
            marker=pack['extra'][6:9].cpu().numpy();rr={}
            for name,mask in regions.items():
                s=marker[:,mask].sum(1,dtype=np.float64);rr[name]=dict(pixels=int(mask.sum()),selected_static_fraction=float(s[0]/s[2]) if s[2] else None,added_static_fraction=float(s[1]/s[2]) if s[2] else None,all_static_alpha_T_mean=float(s[2]/mask.sum()) if mask.any() else None)
            records.append(dict(raw=key,RGB_maxabs_vs_export=err,regions=rr))
    scene.assert_persistent_unchanged()
    photo_equal=True;reference_equal=True
    for frame in m.frames:
        with np.load(a.before/'world_road/views'/f'{frame.key:03d}.npz') as b,np.load(a.after/'world_road/views'/f'{frame.key:03d}.npz') as c:
            photo_equal &= np.array_equal(b['fit'],c['fit']);reference_equal &= np.array_equal(b['reference_linear'],c['reference_linear'])
    assert photo_equal and reference_equal
    receipt=dict(status='PASS_ACTUAL_FIXED_CANDIDATE_AND_REFERENCE_CHECK',records=records,old_candidates=len(old),new_candidates=len(new),added_candidates=len(added),old_candidates_lost=int(len(np.setdiff1d(old,new))),
        all199_photo_weights_identical=bool(photo_equal),all199_reference_linear_identical=bool(reference_equal),parent_unchanged=True,training_updates=0,actual_forward_calls=len(records),automatic_shutdown=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
